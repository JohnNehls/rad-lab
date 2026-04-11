"""Space-Time Adaptive Processing (STAP) for airborne radar clutter rejection.

Simulates a multi-channel (array) pulse-Doppler radar and applies joint
spatial-temporal filtering to separate moving targets from angle-Doppler
coupled ground clutter.  Provides :func:`gen` for end-to-end simulation and
:func:`smi_weights` for the Sample Matrix Inversion adaptive processor.

For analysis (as opposed to data simulation), :func:`clutter_plus_noise_covariance`
builds the theoretical space-time covariance and :func:`sinr_loss` evaluates the
optimal-processor SINR loss, from which the clutter ridge, the minimum
detectable velocity, and the Reed-Mallett-Brennan training-support rule follow.
"""

from __future__ import annotations

import numpy as np
from scipy import fft, signal
import matplotlib.pyplot as plt

from . import constants as c
from .rf_datacube import number_range_bins, range_axis
from .noise import unity_variance_complex_noise
from .uniform_linear_arrays import steering_vector
from ._rdm_internals import _propagation_phase, create_window
from .pulse_doppler_radar import Radar
from .waveform import WaveformSample
from .returns import Target, Return
from .utilities import zero_to_smallest_float


# ---------------------------------------------------------------------------
# Datacube population
# ---------------------------------------------------------------------------


def _add_target_returns(
    datacube: np.ndarray,
    waveform: WaveformSample,
    radar: Radar,
    target: Target,
    el_pos: np.ndarray,
) -> None:
    """Inject a single target's returns into the 3-D datacube.

    For each pulse and each array element the function computes the two-way
    propagation delay (using the target's range and range-rate), applies the
    element-dependent spatial phase from the steering vector, and adds the
    scaled waveform at the appropriate range bin.

    Args:
        datacube: 3-D complex array ``(n_range, n_pulses, n_elements)``,
            modified in place.
        waveform: Waveform containing the discrete pulse samples.
        radar: Radar system parameters.
        target: Target kinematics and RCS.
        el_pos: Element positions normalised by wavelength.
    """
    n_range, n_pulses, n_elements = datacube.shape
    pulse_tx_times = np.arange(n_pulses) / radar.prf

    # Range history (constant range-rate model)
    ranges = target.range + target.range_rate * pulse_tx_times

    # Two-way delay and carrier phase per pulse
    two_way_delays = 2 * ranges / c.C
    two_way_phases = _propagation_phase(two_way_delays, radar.fcar)

    # Range bin indices (within each pulse, not absolute time)
    sample_indices = (
        np.round((two_way_delays - waveform.pulse_width / 2) * radar.sample_rate).astype(int) - 1
    )

    # Target voltage amplitude.  This module works in noise-relative units
    # (unit-variance receiver noise, see ``gen``), so ``sqrt(RCS)`` sets the
    # target's per-sample voltage relative to the noise floor rather than a
    # physical range-equation voltage.  STAP targets must have an RCS.
    amplitude = np.sqrt(target.rcs or 0.0)

    # Steering vector for this target's angle of arrival
    sv = steering_vector(el_pos, target.angle)

    n_wf = len(waveform.pulse_sample)

    for n in range(n_pulses):
        idx = sample_indices[n]
        if idx < 0 or idx + n_wf > n_range:
            continue

        # Baseband pulse with propagation phase
        pulse = amplitude * waveform.pulse_sample * np.exp(1j * two_way_phases[n])

        for e in range(n_elements):
            datacube[idx : idx + n_wf, n, e] += pulse * sv[e]


def _add_clutter(
    datacube: np.ndarray,
    radar: Radar,
    el_pos: np.ndarray,
    platform_velocity: float,
    n_clutter_patches: int = 180,
    cnr: float = 40.0,
) -> None:
    """Add angle-Doppler coupled ground clutter to the datacube.

    Models the ground as discrete isotropic scattering patches distributed
    across all range bins and azimuth angles.  Each patch has a Doppler shift
    determined by its azimuth angle relative to the platform velocity vector,
    producing the characteristic clutter ridge in angle-Doppler space.

    Clutter is injected at every range bin so that the sample covariance
    matrix estimated from training data accurately represents the clutter
    statistics — a requirement for adaptive (STAP) processing.

    Args:
        datacube: 3-D complex array ``(n_range, n_pulses, n_elements)``,
            modified in place.
        radar: Radar system parameters.
        el_pos: Element positions normalised by wavelength.
        platform_velocity: Platform ground speed [m/s].
        n_clutter_patches: Number of azimuth patches to discretise the clutter
            ring.  More patches give a smoother clutter ridge.
        cnr: Aggregate clutter-to-noise power ratio [linear], summed over all
            patches, injected at each range bin.  With the unit-variance noise
            used by :func:`gen` this is an honest clutter-to-noise ratio: the
            per-patch amplitude ``sqrt(cnr / n_clutter_patches)`` sums (over
            random-phase patches) to a clutter power of ``cnr`` per sample.
    """
    n_range, n_pulses, _ = datacube.shape

    # Clutter angles uniformly distributed in azimuth
    angles = np.linspace(-90, 90, n_clutter_patches, endpoint=False)

    wavelength = c.C / radar.fcar

    # Amplitude per patch per range bin
    patch_amplitude = np.sqrt(cnr / n_clutter_patches)

    for angle in angles:
        # Doppler shift from platform motion
        fd = 2 * platform_velocity * np.sin(np.deg2rad(angle)) / wavelength

        # Spatial steering vector for this clutter angle
        sv = steering_vector(el_pos, angle)

        # Per-pulse Doppler phase ramp
        n_vec = np.arange(n_pulses)
        doppler_phasor = np.exp(1j * 2 * np.pi * fd * n_vec / radar.prf)

        # Inject clutter at every range bin with a random phase per bin
        # (different scatterer realisations at each range).  The contribution
        # separates into a range term, a pulse (Doppler) term, and an element
        # (spatial) term, so broadcast the outer product across the whole cube.
        random_phases = np.exp(1j * 2 * np.pi * np.random.rand(n_range))

        datacube += (
            patch_amplitude
            * random_phases[:, None, None]
            * doppler_phasor[None, :, None]
            * sv[None, None, :]
        )


# ---------------------------------------------------------------------------
# STAP processing
# ---------------------------------------------------------------------------


def _space_time_snapshot(
    datacube: np.ndarray,
    range_bin: int,
) -> np.ndarray:
    """Extract space-time snapshots for a single range bin.

    The snapshot vector for range bin *k* is formed by stacking the
    pulse × element slice into a single column vector of length
    ``n_pulses * n_elements``.

    Args:
        datacube: 3-D array ``(n_range, n_pulses, n_elements)``.
        range_bin: Range bin index.

    Returns:
        1-D complex vector of length ``n_pulses * n_elements``.
    """
    # datacube[k] has shape (n_pulses, n_elements)
    return datacube[range_bin].flatten()


def _covariance_matrix(
    datacube: np.ndarray,
    range_bin: int,
    n_guard: int = 5,
) -> np.ndarray:
    """Estimate the clutter-plus-noise covariance from training range bins.

    Uses secondary data (range bins away from the cell under test) to form
    the sample covariance matrix.

    Args:
        datacube: 3-D array ``(n_range, n_pulses, n_elements)``.
        range_bin: Index of the cell under test (excluded from training).
        n_guard: Number of guard bins on each side of the CUT to exclude.
            Must exceed the target's range-compressed extent (roughly the
            pulse length in samples): a strong target's compression sidelobes
            reaching into the training data make the covariance "learn" the
            target, and STAP then nulls it (self-cancellation).

    Returns:
        Sample covariance matrix of shape ``(NM, NM)`` where
        ``NM = n_pulses * n_elements``.
    """
    n_range = datacube.shape[0]
    NM = datacube.shape[1] * datacube.shape[2]

    R = np.zeros((NM, NM), dtype=complex)
    count = 0

    for k in range(n_range):
        if abs(k - range_bin) <= n_guard:
            continue
        x = datacube[k].flatten()
        R += np.outer(x, np.conj(x))
        count += 1

    if count > 0:
        R /= count

    return R


def smi_weights(
    R: np.ndarray,
    steering_vec: np.ndarray,
    diagonal_load: float = 0.0,
) -> np.ndarray:
    """Compute adaptive STAP weights via Sample Matrix Inversion (SMI).

    The optimal weight vector maximises the output SINR for a target with
    the given space-time steering vector:

    .. math::

        \\mathbf{w} = \\mathbf{R}^{-1} \\mathbf{s}

    where **R** is the clutter-plus-noise covariance and **s** is the
    space-time steering vector.

    Args:
        R: Clutter-plus-noise covariance matrix ``(NM, NM)``.
        steering_vec: Space-time steering vector ``(NM,)`` for the desired
            target angle and Doppler.
        diagonal_load: Optional diagonal loading factor for numerical
            stability.  Added as ``diagonal_load * I`` to **R** before
            inversion.

    Returns:
        Adaptive weight vector ``(NM,)``.
    """
    NM = R.shape[0]
    R_loaded = R + diagonal_load * np.eye(NM)
    w = np.linalg.solve(R_loaded, steering_vec)
    return w


def space_time_steering_vector(
    el_pos: np.ndarray,
    n_pulses: int,
    angle: float,
    fd: float,
    prf: float,
) -> np.ndarray:
    """Build the space-time steering vector for a given angle and Doppler.

    The space-time steering vector is the Kronecker product of the temporal
    steering vector (Doppler) and the spatial steering vector (angle):

    .. math::

        \\mathbf{s} = \\mathbf{a}_t \\otimes \\mathbf{a}_s

    Args:
        el_pos: Element positions normalised by wavelength.
        n_pulses: Number of pulses in the CPI.
        angle: Target angle of arrival [degrees], 0 = broadside.
        fd: Target Doppler frequency [Hz].
        prf: Pulse repetition frequency [Hz].

    Returns:
        Space-time steering vector of length ``n_pulses * n_elements``.
    """
    # Spatial steering vector
    a_s = steering_vector(el_pos, angle)

    # Temporal steering vector
    n_vec = np.arange(n_pulses)
    a_t = np.exp(1j * 2 * np.pi * fd * n_vec / prf)

    # Kronecker product: temporal ⊗ spatial
    return np.kron(a_t, a_s)


# ---------------------------------------------------------------------------
# Analysis: theoretical covariance, clutter ridge, and SINR loss
# ---------------------------------------------------------------------------


def clutter_plus_noise_covariance(
    el_pos: np.ndarray,
    n_pulses: int,
    radar: Radar,
    platform_velocity: float,
    cnr: float = 40.0,
    n_clutter_patches: int = 180,
) -> np.ndarray:
    """Build the theoretical space-time clutter-plus-noise covariance.

    Sums the outer products of the space-time steering vectors of all clutter
    patches (weighted by per-patch power) and adds unit-variance white noise:

    .. math::

        \\mathbf{R} = \\sum_i \\frac{\\text{cnr}}{N_\\text{patch}}
        \\mathbf{v}_i \\mathbf{v}_i^H + \\mathbf{I}

    where :math:`\\mathbf{v}_i` is the space-time steering vector of patch *i*
    at its azimuth angle and platform-induced Doppler.  This is the *ensemble*
    covariance the finite-sample estimate in :func:`_covariance_matrix`
    converges to; being noise-free it gives clean clutter-ridge and SINR-loss
    curves for analysis (as opposed to a noisy data estimate).

    Args:
        el_pos: Element positions normalised by wavelength.
        n_pulses: Number of pulses in the CPI.
        radar: Radar system parameters.
        platform_velocity: Platform ground speed [m/s].
        cnr: Aggregate clutter-to-noise power ratio (see :func:`_add_clutter`).
        n_clutter_patches: Number of azimuth patches for the clutter ring.

    Returns:
        ``(NM, NM)`` Hermitian covariance in noise-relative units (noise = I),
        with ``NM = n_pulses * n_elements``.
    """
    wavelength = c.C / radar.fcar
    angles = np.linspace(-90, 90, n_clutter_patches, endpoint=False)
    patch_power = cnr / n_clutter_patches

    NM = n_pulses * len(el_pos)
    R = np.eye(NM, dtype=complex)  # unit-variance white noise floor

    for angle in angles:
        fd = 2 * platform_velocity * np.sin(np.deg2rad(angle)) / wavelength
        v = space_time_steering_vector(el_pos, n_pulses, angle, fd, radar.prf)
        R += patch_power * np.outer(v, np.conj(v))

    return R


def sinr_loss(R: np.ndarray, steering_vec: np.ndarray) -> float:
    """Normalised SINR loss of the optimal processor for one steering vector.

    Ratio of the clutter-limited optimal SINR to the clutter-free (noise-only)
    matched-filter SNR:

    .. math::

        L = \\frac{\\text{SINR}_\\text{opt}}{\\text{SNR}_0}
          = \\frac{\\mathbf{s}^H \\mathbf{R}^{-1} \\mathbf{s}}{\\mathbf{s}^H \\mathbf{s}}

    with **R** in noise-relative units (white-noise level 1).  ``L = 1`` (0 dB)
    means no loss (clutter-free); a deep notch (large negative dB) marks the
    Dopplers masked by the clutter ridge and sets the minimum detectable
    velocity (MDV).

    Args:
        R: Clutter-plus-noise covariance ``(NM, NM)`` (noise = I).
        steering_vec: Space-time steering vector ``(NM,)``.

    Returns:
        SINR loss as a linear ratio in ``(0, 1]``.
    """
    Rinv_s = np.linalg.solve(R, steering_vec)
    return float(
        np.real(np.vdot(steering_vec, Rinv_s)) / np.real(np.vdot(steering_vec, steering_vec))
    )


# ---------------------------------------------------------------------------
# Conventional (non-adaptive) processing
# ---------------------------------------------------------------------------


def conventional_rdm(
    datacube: np.ndarray,
    radar: Radar,
    el_pos: np.ndarray,
    steer_angle: float = 0.0,
    window: str = "chebyshev",
    window_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a 3-D datacube with conventional beamforming + Doppler FFT.

    Applies spatial beamforming (steering vector dot product across elements)
    to collapse the array dimension, then windows and Doppler-processes the
    result to produce a standard 2-D range-Doppler map.

    Args:
        datacube: 3-D complex array ``(n_range, n_pulses, n_elements)``.
        radar: Radar system parameters.
        el_pos: Element positions normalised by wavelength.
        steer_angle: Beamforming steering angle [degrees].
        window: Doppler window function (same options as ``rdm.gen``).
        window_kwargs: Optional window parameters.

    Returns:
        tuple: ``(rdot_axis, r_axis, rdm_out)``:

            - **rdot_axis** (*np.ndarray*): Range-rate axis [m/s].
            - **r_axis** (*np.ndarray*): Range axis [m].
            - **rdm_out** (*np.ndarray*): 2-D range-Doppler map.
    """
    n_range, n_pulses, n_elements = datacube.shape

    # Beamform: dot product with steering vector across elements
    sv = steering_vector(el_pos, steer_angle)
    beamformed = np.zeros((n_range, n_pulses), dtype=complex)
    for e in range(n_elements):
        beamformed += np.conj(sv[e]) * datacube[:, :, e]

    # Doppler window
    win_mat = create_window(
        beamformed.shape, window=window, window_kwargs=window_kwargs, plot=False
    )
    beamformed *= win_mat

    # Doppler FFT
    prf = radar.sample_rate / n_range
    f_axis = fft.fftshift(fft.fftfreq(n_pulses, 1 / prf))
    beamformed[:] = fft.fftshift(fft.fft(beamformed, axis=1), axes=1)

    r_ax = range_axis(radar.sample_rate, n_range)
    rdot_axis = -c.C * f_axis / (2 * radar.fcar)

    return rdot_axis, r_ax, beamformed


# ---------------------------------------------------------------------------
# Adaptive processing
# ---------------------------------------------------------------------------


def adaptive_rdm(
    datacube: np.ndarray,
    radar: Radar,
    el_pos: np.ndarray,
    steer_angle: float = 0.0,
    n_guard: int = 5,
    diagonal_load: float = 1.0,
    window: str = "chebyshev",
    window_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a 3-D datacube with STAP (Sample Matrix Inversion).

    For each range bin and each Doppler frequency, builds a space-time
    steering vector, estimates the clutter covariance from training data,
    and forms the adaptive matched filter statistic
    ``(s^H R^-1 x) / sqrt(s^H R^-1 s)``.  The ``sqrt(s^H R^-1 s)``
    normalisation flattens the residual noise floor across the map so the
    output is a detection statistic in which the target stands out (the raw
    ``w^H x`` has an angle-Doppler-dependent gain that buries the target).

    A Doppler taper is applied to the temporal part of the steering vector,
    matching the window used in conventional processing.  Without it a strong
    target rings out across Doppler at the rectangular-taper sidelobe level
    (~-13 dB), smearing it into a horizontal streak in the range-Doppler map.

    Args:
        datacube: 3-D complex array ``(n_range, n_pulses, n_elements)``.
        radar: Radar system parameters.
        el_pos: Element positions normalised by wavelength.
        steer_angle: Look-direction steering angle [degrees].
        n_guard: Guard bins for covariance estimation.  Should exceed the
            target's range-compressed extent (~ the pulse length in samples)
            to avoid target self-cancellation; see :func:`_covariance_matrix`.
        diagonal_load: Diagonal loading factor for matrix inversion.  In the
            noise-relative units of :func:`gen` (unit-variance noise) this is
            a small multiple of the noise floor.
        window: Doppler taper for the steering vector (same options as
            :func:`gen`).
        window_kwargs: Optional window parameters.

    Returns:
        tuple: ``(rdot_axis, r_axis, rdm_out)``:

            - **rdot_axis** (*np.ndarray*): Range-rate axis [m/s].
            - **r_axis** (*np.ndarray*): Range axis [m].
            - **rdm_out** (*np.ndarray*): 2-D adaptively filtered RDM.
    """
    n_range, n_pulses, n_elements = datacube.shape
    prf = radar.sample_rate / n_range

    f_axis = fft.fftshift(fft.fftfreq(n_pulses, 1 / prf))
    rdot_axis = -c.C * f_axis / (2 * radar.fcar)
    r_ax = range_axis(radar.sample_rate, n_range)

    # Space-time Doppler taper: window the temporal (pulse) dimension, leave
    # the spatial (element) dimension uniform.  Matches conventional windowing.
    doppler_taper = create_window(
        (1, n_pulses), window=window, window_kwargs=window_kwargs, plot=False
    ).ravel()
    st_taper = np.kron(doppler_taper, np.ones(n_elements))

    rdm_out = np.zeros((n_range, n_pulses), dtype=complex)

    for k in range(n_range):
        # Estimate covariance from training data
        R = _covariance_matrix(datacube, k, n_guard=n_guard)

        x = _space_time_snapshot(datacube, k)

        for m, fd in enumerate(f_axis):
            # Build the (tapered) space-time steering vector for this angle
            # and Doppler
            s = space_time_steering_vector(el_pos, n_pulses, steer_angle, fd, prf) * st_taper

            # Adaptive weights w = R^-1 s (Sample Matrix Inversion)
            w = smi_weights(R, s, diagonal_load=diagonal_load)

            # Adaptive matched filter statistic: normalise by sqrt(s^H R^-1 s).
            # This flattens the residual noise floor across the map so the
            # target stands out.  The unnormalised w^H x has a steering-vector-
            # dependent gain that varies over angle-Doppler and buries the
            # target in the residual (its magnitude is not a detection statistic).
            snr_norm = np.sqrt(np.real(np.vdot(s, w)))
            rdm_out[k, m] = np.dot(np.conj(w), x) / snr_norm

    return rdot_axis, r_ax, rdm_out


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------


def gen(
    radar: Radar,
    waveform: WaveformSample,
    return_list: list[Return],
    el_pos: np.ndarray,
    platform_velocity: float = 0.0,
    cnr: float = 0.0,
    n_clutter_patches: int = 180,
    steer_angle: float = 0.0,
    seed: int = 0,
    plot: bool = True,
    window: str = "chebyshev",
    window_kwargs: dict | None = None,
    n_guard: int = 5,
    diagonal_load: float = 1.0,
) -> dict:
    """Generate a multi-channel RDM and process with conventional and STAP filters.

    Simulates a ULA pulse-Doppler radar with ground clutter and noise, then
    produces both a conventional beamformed RDM and an adaptively filtered
    (STAP) RDM for comparison.

    Each :class:`~rad_lab.returns.Target` in *return_list* must have an
    ``angle`` attribute giving its angle of arrival in degrees (0 = broadside).
    This can be set as ``Target(range=..., range_rate=..., rcs=..., angle=...)``.

    Because STAP is governed by power *ratios* (CNR, SNR, SINR) rather than
    absolute power, this module works in noise-relative units: the receiver
    noise is unit-variance complex Gaussian, and target (``sqrt(RCS)``) and
    clutter (``sqrt(cnr)``) voltages are referenced to it.

    Args:
        radar: Radar system parameters.
        waveform: Waveform to transmit.
        return_list: List of :class:`~rad_lab.returns.Return` objects.
        el_pos: Array element positions normalised by wavelength.
        platform_velocity: Platform ground speed [m/s].  Determines the
            clutter Doppler coupling.  Set to 0 for no clutter Doppler.
        cnr: Aggregate clutter-to-noise power ratio [linear] at each range
            bin (summed over all patches).  Set to 0 to disable clutter.
        n_clutter_patches: Number of azimuth patches for clutter modelling.
        steer_angle: Beamforming / STAP look direction [degrees].
        seed: Random seed for reproducibility.
        plot: If True, plot conventional and adaptive RDMs side by side.
        window: Doppler window function.
        window_kwargs: Optional window parameters.
        n_guard: Guard bins for STAP covariance estimation.  Should exceed the
            target's range-compressed extent (~ the pulse length in samples)
            to avoid target self-cancellation; see :func:`_covariance_matrix`.
        diagonal_load: Diagonal loading for STAP matrix inversion (a small
            multiple of the unit-variance noise floor).

    Returns:
        dict with keys:

            - ``"rdot_axis"``: Range-rate axis [m/s].
            - ``"r_axis"``: Range axis [m].
            - ``"datacube"``: Raw 3-D datacube after range compression.
            - ``"conventional"``: 2-D conventional RDM.
            - ``"adaptive"``: 2-D STAP-processed RDM.
    """
    np.random.seed(seed)

    waveform.set_sample(radar.sample_rate)
    n_range = number_range_bins(radar.sample_rate, radar.prf)
    n_elements = len(el_pos)

    # Allocate 3-D datacube: range × pulses × elements
    datacube = np.zeros((n_range, radar.n_pulses, n_elements), dtype=np.complex64)

    # Inject target returns
    for ret in return_list:
        _add_target_returns(datacube, waveform, radar, ret.target, el_pos)

    # Inject clutter
    if cnr > 0:
        _add_clutter(
            datacube,
            radar,
            el_pos,
            platform_velocity,
            n_clutter_patches,
            cnr,
        )

    # Add receiver noise.  STAP is governed by *ratios* (CNR, SNR, SINR), not
    # absolute power, so this module works in noise-relative units: unit-variance
    # complex Gaussian noise, with target (sqrt(RCS)) and clutter (sqrt(cnr))
    # amplitudes referenced to it.  This keeps ``cnr`` an honest clutter-to-noise
    # ratio and gives ``diagonal_load`` a meaningful scale (a small multiple of
    # the unit noise floor).  Complex noise — not the real-valued draw used
    # previously — matches the I/Q statistics assumed by the covariance estimate.
    datacube += unity_variance_complex_noise(datacube.shape)

    # Range compression (per element)
    kernel = np.conj(waveform.pulse_sample)[::-1]
    for e in range(n_elements):
        dc_2d = datacube[:, :, e]
        dc_2d[:] = signal.fftconvolve(dc_2d, kernel.reshape(-1, 1), mode="same", axes=0)

    # Conventional processing
    rdot_axis, r_axis, conv_rdm = conventional_rdm(
        datacube.copy(),
        radar,
        el_pos,
        steer_angle,
        window,
        window_kwargs,
    )

    # Adaptive (STAP) processing
    _, _, adapt_rdm = adaptive_rdm(
        datacube.copy(),
        radar,
        el_pos,
        steer_angle,
        n_guard,
        diagonal_load,
        window,
        window_kwargs,
    )

    if plot:
        plot_comparison(rdot_axis, r_axis, conv_rdm, adapt_rdm)

    return {
        "rdot_axis": rdot_axis,
        "r_axis": r_axis,
        "datacube": datacube,
        "conventional": conv_rdm,
        "adaptive": adapt_rdm,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(
    rdot_axis: np.ndarray,
    r_axis: np.ndarray,
    conv_rdm: np.ndarray,
    adapt_rdm: np.ndarray,
    cbar_min: float = -60,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot conventional and adaptive RDMs side by side.

    Args:
        rdot_axis: Range-rate axis [m/s].
        r_axis: Range axis [m].
        conv_rdm: 2-D conventional RDM.
        adapt_rdm: 2-D STAP-processed RDM.
        cbar_min: Minimum colorbar value [dB].

    Returns:
        The figure and a tuple of the two axes.
    """
    fig, (ax_conv, ax_stap) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Conventional vs. STAP Processing")

    for ax, data, title in [
        (ax_conv, conv_rdm, "Conventional (Beamform + FFT)"),
        (ax_stap, adapt_rdm, "STAP (SMI)"),
    ]:
        magnitude = np.abs(data)
        zero_to_smallest_float(magnitude)
        plot_data = 20 * np.log10(magnitude / magnitude.max())

        mesh = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, plot_data, shading="auto")
        mesh.set_clim(cbar_min, 0)
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Normalised Magnitude [dB]")
        ax.set_title(title)
        ax.set_xlabel("Range Rate [km/s]")
        ax.set_ylabel("Range [km]")

    fig.tight_layout()
    return fig, (ax_conv, ax_stap)
