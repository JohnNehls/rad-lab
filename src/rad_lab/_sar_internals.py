"""SAR raw-data generation and azimuth compression internals.

Provides the low-level functions that populate a SAR datacube with point-target
returns and focus the data in the cross-range (azimuth) dimension.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.linalg import norm
from scipy import fft

from . import constants as c
from .geometry import slant_range
from ._rdm_internals import _propagation_phase, _return_sample_indices, _inject_pulses
from .waveform import WaveformSample
from .sar_radar import SarRadar, SarTarget


def _beam_weights(
    platform_positions: np.ndarray,
    target_position: list[float],
    scene_center: list[float] | np.ndarray,
    beamwidth: float,
    pattern: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Computes per-pulse two-way antenna gain weights for a steered beam.

    For each pulse the antenna boresight points from the platform toward
    ``scene_center``.  The scalar off-boresight angle to ``target_position``
    is computed, and a two-way beam pattern is applied.  The pattern is
    radially symmetric — it depends only on the magnitude of the
    off-boresight angle, with no distinction between azimuth and elevation.

    Args:
        platform_positions: Platform positions ``(n_pulses, 3)`` [m].
        target_position: Target ``[x, y, z]`` [m].
        scene_center: Boresight reference point(s).  Shape ``(3,)`` for a
            fixed point (spotlight) or ``(n_pulses, 3)`` for per-pulse
            broadside points (stripmap).
        beamwidth: One-way 3-dB beamwidth [rad].
        pattern: Optional callable that maps an array of off-boresight
            angles [rad] to two-way amplitude weights.  Defaults to a
            Gaussian: ``exp(-4 ln2 (θ / beamwidth)²)``.

    Returns:
        Per-pulse amplitude weights with shape ``(n_pulses,)``.
    """
    sc = np.asarray(scene_center)
    tp = np.asarray(target_position)

    # Vectors from each platform position to scene centre and to target
    to_scene = sc - platform_positions  # (n_pulses, 3)
    to_target = tp - platform_positions  # (n_pulses, 3)

    # Norms
    d_scene = norm(to_scene, axis=1)
    d_target = norm(to_target, axis=1)

    # Cosine of the angle between the two direction vectors
    cos_theta = np.sum(to_scene * to_target, axis=1) / (d_scene * d_target)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if pattern is not None:
        return pattern(theta)

    # Default: two-way Gaussian (-3 dB at theta = beamwidth/2, squared for two-way)
    return np.exp(-4 * np.log(2) * (theta / beamwidth) ** 2)


def add_sar_returns(
    datacube: np.ndarray,
    waveform: WaveformSample,
    sar_radar: SarRadar,
    target_list: list[SarTarget],
    platform_positions: np.ndarray,
    beam_weights_fn: Callable[[list[float]], np.ndarray] | None = None,
) -> None:
    """Populates a datacube with SAR point-target returns.

    For each pulse position and each target the function computes the slant
    range, two-way propagation delay and carrier phase, then injects the
    waveform at the corresponding range bin.  The datacube is modified
    in place.

    In spotlight mode, ``beam_weights_fn`` supplies per-pulse amplitude
    weights that model the steered antenna beam pattern.

    Args:
        datacube: 2-D complex array of shape ``(n_range_bins, n_pulses)``.
        waveform: Waveform containing the discrete pulse samples.
        sar_radar: SAR system parameters.
        target_list: List of :class:`SarTarget` point scatterers.
        platform_positions: Platform positions ``(n_pulses, 3)`` [m].
        beam_weights_fn: Optional callable that accepts a target position
            ``[x, y, z]`` and returns per-pulse amplitude weights
            ``(n_pulses,)``.  Used for spotlight beam-pattern weighting.
    """
    n_pulses = datacube.shape[1]
    pulse_tx_times = np.arange(n_pulses) / sar_radar.prf

    for target in target_list:
        # Slant range from every aperture position to this target [m]
        ranges = slant_range(platform_positions, target.position)

        # Two-way propagation delay and carrier phase per pulse
        two_way_delays = 2 * ranges / c.C
        two_way_phases = _propagation_phase(two_way_delays, sar_radar.fcar)

        # Absolute return times (pulse tx time + two-way delay) are needed so
        # that each pulse's waveform lands in the correct section of the flat array.
        return_times = pulse_tx_times + two_way_delays
        sample_indices = _return_sample_indices(return_times, waveform, sar_radar.sample_rate)

        # Amplitude: sqrt(RCS), optionally weighted by beam pattern
        amplitude: float | np.ndarray = np.sqrt(target.rcs)
        if beam_weights_fn is not None:
            amplitude = amplitude * beam_weights_fn(target.position)

        _inject_pulses(
            datacube,
            waveform.pulse_sample,
            sample_indices,
            two_way_phases,
            amplitude=amplitude,
        )


def azimuth_matched_filter(
    datacube: np.ndarray,
    sar_radar: SarRadar,
    range_axis: np.ndarray,
) -> np.ndarray:
    """Focuses a range-compressed datacube in the azimuth (cross-range) dimension.

    For each range bin the function builds a reference phase history based on
    the exact hyperbolic slant-range variation across the synthetic aperture,
    then correlates it with the data via FFT convolution along the slow-time
    axis.

    Args:
        datacube: 2-D complex array ``(n_range_bins, n_pulses)``, already
            range-compressed.
        sar_radar: SAR system parameters.
        range_axis: 1-D range axis [m] with length ``n_range_bins``.

    Returns:
        Cross-range (azimuth) axis [m] with length ``n_pulses``.
    """
    n_range_bins, n_pulses = datacube.shape

    # Along-track positions of each aperture sample, centred at 0
    along_track = (np.arange(n_pulses) - n_pulses / 2) * sar_radar.pulse_spacing

    for k in range(n_range_bins):
        R0 = range_axis[k]

        # Exact hyperbolic range history for a target at broadside range R0
        R_history = np.sqrt(R0**2 + along_track**2)

        # Reference signal: the phase a broadside target would produce
        h_ref = np.exp(-1j * 4 * np.pi / sar_radar.wavelength * R_history)

        # Matched filter = correlation via FFT: ifft(FFT(data) * conj(FFT(h_ref)))
        datacube[k, :] = fft.fftshift(fft.ifft(fft.fft(datacube[k, :]) * np.conj(fft.fft(h_ref))))

    # Cross-range axis maps directly to along-track position
    cross_range_axis = along_track.copy()

    return cross_range_axis


def rcmc(
    datacube: np.ndarray,
    sar_radar: SarRadar,
    range_axis: np.ndarray,
    debug: bool = False,
) -> None:
    """Range Cell Migration Correction (RCMC) for the Range-Doppler Algorithm.

    A target at closest-approach range ``R0`` traces a hyperbolic trajectory
    in slant range across the synthetic aperture: ``R(eta) = sqrt(R0^2 +
    v^2 eta^2)``.  After range compression the target's energy is therefore
    spread across several range bins as a function of slow-time, which
    smears the azimuth-compressed peak unless the trajectory is realigned
    to a constant range first.

    In the range-Doppler domain (azimuth FFT applied), migration as a
    function of azimuth Doppler frequency ``f_eta`` collapses to a closed
    form per range bin::

        dR(f_eta, R0) = R0 * (1 / sqrt(1 - (lambda f_eta / (2 v))^2) - 1)

    This function shifts each range column by ``-dR / dR_grid`` bins
    (sub-bin precision via 8-tap Hann-windowed sinc interpolation), in
    place.  After RCMC, :func:`azimuth_matched_filter` correlates with
    its exact hyperbolic reference and produces a sharp PSF.

    Args:
        datacube: 2-D complex array ``(n_range_bins, n_pulses)``, already
            range-compressed.  Modified in place.
        sar_radar: SAR system parameters.  Provides ``wavelength``,
            ``platform_velocity``, ``prf``, and ``sample_rate``.
        range_axis: 1-D range axis [m] with length ``n_range_bins``.
        debug: If True, plot the range-Doppler map (RDM) before and
            after the correction.  The "before" panel shows curved
            migration trajectories; the "after" panel shows them
            collapsed to horizontal lines at each target's R0.

    Returns:
        None.  ``datacube`` is modified in place.
    """
    n_range_bins, n_pulses = datacube.shape

    # Range bin spacing [m]: two-way travel between adjacent fast-time samples
    dR_grid = c.C / (2 * sar_radar.sample_rate)

    # Azimuth Doppler frequency axis [Hz], in unshifted fft order so it
    # aligns with the result of fft.fft along slow-time without an fftshift.
    f_eta = fft.fftfreq(n_pulses, d=1.0 / sar_radar.prf)

    # Doppler argument lambda * f_eta / (2 v).  Clipped defensively to keep
    # sqrt() real near the 2v/lambda divergence (unreachable for sane SAR
    # configs, but a student running unusual parameters won't get NaNs).
    doppler_arg = sar_radar.wavelength * f_eta / (2.0 * sar_radar.platform_velocity)
    safe_arg = np.clip(doppler_arg, -0.999, 0.999)
    inv_cos_factor = 1.0 / np.sqrt(1.0 - safe_arg**2) - 1.0  # (n_pulses,)

    # 8-tap kernel offsets t = -3..4.  The desired sub-bin location after
    # base_idx + frac falls between taps 0 and 1, so the kernel midpoint
    # in tap-value coordinates is 0.5 (not the array index midpoint).
    kernel_taps = np.arange(-3, 5)
    # Hann window over the 8-tap support, symmetric about kernel midpoint 0.5
    kernel_window = 0.5 * (1.0 + np.cos(np.pi * (kernel_taps - 0.5) / 4.0))  # (8,)

    # Transform once into the range-Doppler domain
    datacube[:] = fft.fft(datacube, axis=1)

    if debug:
        # Lazy import: keeps matplotlib off the import path of
        # _sar_internals when not plotting.
        from .sar import _plot_rdm

        _plot_rdm(range_axis, f_eta, datacube, "RDM before RCMC")

    col_indices = np.arange(n_pulses)[np.newaxis, :]
    for k in range(n_range_bins):
        R0 = range_axis[k]

        # Per-Doppler shift in range bins; positive => trajectory at this
        # Doppler is farther than R0, so we pull data from a larger row.
        delta_bins = R0 * inv_cos_factor / dR_grid  # (n_pulses,)
        base_idx = np.floor(delta_bins).astype(np.int64)  # (n_pulses,)
        frac = delta_bins - base_idx  # (n_pulses,) in [0, 1)

        # Source rows for the 8 taps: shape (8, n_pulses)
        src_rows = k + base_idx[np.newaxis, :] + kernel_taps[:, np.newaxis]
        in_bounds = (src_rows >= 0) & (src_rows < n_range_bins)
        clipped_rows = np.where(in_bounds, src_rows, 0)

        gathered = datacube[clipped_rows, col_indices]  # (8, n_pulses)
        gathered = np.where(in_bounds, gathered, 0.0)

        # Hann-windowed sinc weights at sub-bin offset, normalised so a
        # constant input is reproduced (DC gain = 1 for any sub-bin shift).
        t_minus_frac = kernel_taps[:, np.newaxis] - frac[np.newaxis, :]  # (8, n_pulses)
        weights = np.sinc(t_minus_frac) * kernel_window[:, np.newaxis]
        weights /= np.sum(weights, axis=0, keepdims=True)

        datacube[k, :] = np.sum(weights * gathered, axis=0)

    if debug:
        from .sar import _plot_rdm

        _plot_rdm(range_axis, f_eta, datacube, "RDM after RCMC")

    # Back to slow-time so downstream stages see the expected domain
    datacube[:] = fft.ifft(datacube, axis=1)
