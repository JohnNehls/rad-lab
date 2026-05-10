"""SAR image generation and plotting (stripmap and spotlight modes).

Provides the :func:`gen` entry point that simulates a full synthetic aperture —
transmitting pulses along a straight flight path, injecting point-target
returns, range-compressing, and azimuth-focusing to produce a SAR image.
Spotlight mode is activated by setting ``scene_center`` and ``beamwidth``
on the :class:`~rad_lab.sar_radar.SarRadar` instance.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from . import constants as c
from .rf_datacube import number_range_bins, range_axis, data_cube, matchfilter
from .range_equation import noise_power
from .utilities import zero_to_smallest_float
from ._rdm_internals import create_window
from ._sar_internals import (
    add_sar_returns,
    azimuth_matched_filter,
    rcmc as apply_rcmc,
    _beam_weights,
)
from .geometry import flight_path
from .sar_radar import SarRadar, SarTarget
from .waveform import WaveformSample


def gen(
    sar_radar: SarRadar,
    waveform: WaveformSample,
    target_list: list[SarTarget],
    seed: int = 0,
    plot: bool = True,
    debug: bool = False,
    window: str = "chebyshev",
    window_kwargs: dict | None = None,
    rcmc: bool = True,
    beam_pattern: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a focused SAR image from point-target returns.

    Simulates a SAR collection over a straight, level flight path.  For each
    aperture position the function computes range and phase to every target,
    injects the waveform, then processes the datacube through range
    compression, azimuth windowing, and azimuth matched-filter focusing.

    **Stripmap** mode (default): leave ``sar_radar.scene_center`` as
    ``None``.  Optionally set ``sar_radar.beamwidth`` to apply
    broadside beam-pattern weighting (body-fixed antenna).

    **Spotlight** mode: set both fields on the :class:`SarRadar` instance.
    The antenna beam is steered toward ``scene_center`` each pulse, and
    target amplitudes are weighted by a two-way Gaussian beam pattern.

    Args:
        sar_radar: SAR system parameters.
            See :class:`rad_lab.sar_radar.SarRadar`.
        waveform: WaveformSample created by a factory function
            (e.g. :func:`rad_lab.waveform.lfm_waveform`).
        target_list: List of :class:`~rad_lab.sar_radar.SarTarget` point
            scatterers.
        seed: Random number generator seed for reproducibility.
        plot: If True, plots the focused SAR image.
        debug: If True, plots intermediate processing steps (raw data,
            range-compressed data).
        window: Window function applied along the azimuth dimension before
            focusing.  One of ``"chebyshev"`` (default),
            ``"blackman-harris"``, ``"taylor"``, or ``"none"``.
        window_kwargs: Optional dict forwarded to the window function.
            See :func:`._rdm_internals.create_window`.
        rcmc: If True (default), apply Range Cell Migration Correction
            after range compression and before azimuth focusing.  Disable
            to study the effect of uncorrected range migration on the
            focused PSF (azimuth peak position will tilt across range
            bins, especially at long aperture or large slant range).
        beam_pattern: Optional callable that maps off-boresight angles
            [rad] to amplitude weights.  Overrides the default Gaussian
            in spotlight mode.  See
            :func:`~rad_lab.uniform_linear_arrays.ula_pattern` for a
            convenient way to build one from a ULA specification.

    Returns:
        tuple: ``(cross_range_axis, r_axis, focused_dc)``:

            - **cross_range_axis** (*np.ndarray*): 1-D cross-range axis [m].
            - **r_axis** (*np.ndarray*): 1-D slant-range axis [m].
            - **focused_dc** (*np.ndarray*): 2-D focused SAR image (signal + noise).
    """
    np.random.seed(seed)

    ########## Waveform setup #####################################################################
    waveform.set_sample(sar_radar.sample_rate)

    ########## Flight path ########################################################################
    platform_positions = flight_path(
        sar_radar.n_pulses, sar_radar.pulse_spacing, sar_radar.platform_altitude
    )

    ########## Create datacube and range axis ######################################################
    n_range_bins = number_range_bins(sar_radar.sample_rate, sar_radar.prf)
    r_axis = range_axis(sar_radar.sample_rate, n_range_bins)

    datacube = data_cube(sar_radar.sample_rate, sar_radar.prf, sar_radar.n_pulses)

    ########## Populate with target returns ########################################################
    beam_weights_fn = None
    if sar_radar.scene_center is not None:
        # Spotlight: beam steered toward a fixed scene centre
        beam_weights_fn = partial(
            _beam_weights,
            platform_positions,
            scene_center=sar_radar.scene_center,
            beamwidth=sar_radar.beamwidth,
            pattern=beam_pattern,
        )
    elif sar_radar.beamwidth is not None:
        # Stripmap with beam weighting: body-fixed antenna pointing broadside
        def _stripmap_beam_fn(target_position: list[float]) -> np.ndarray:
            broadside = platform_positions.copy()
            broadside[:, 1] = target_position[1]  # target's cross-track distance
            broadside[:, 2] = 0.0
            return _beam_weights(
                platform_positions,
                target_position,
                scene_center=broadside,
                beamwidth=sar_radar.beamwidth,
                pattern=beam_pattern,
            )

        beam_weights_fn = _stripmap_beam_fn

    add_sar_returns(
        datacube, waveform, sar_radar, target_list, platform_positions, beam_weights_fn
    )

    ########## Add noise ##########################################################################
    # PSD-based: σ² = R · N₀ · fs, so a B-wide band has variance R · N₀ · B.
    rx_noise_volt = np.sqrt(
        c.RADAR_LOAD
        * noise_power(sar_radar.sample_rate, sar_radar.noise_factor, sar_radar.op_temp)
    )
    noise_dc = np.random.uniform(low=-1, high=1, size=datacube.shape) * rx_noise_volt

    datacube += noise_dc
    del noise_dc

    if debug:
        _plot_raw(r_axis, datacube.real, "Raw SAR data (real)")

    ########## Range compression ##################################################################
    matchfilter(datacube, waveform.pulse_sample, pedantic=False)

    if debug:
        _plot_raw(r_axis, datacube, "Range-compressed")

    ########## Range Cell Migration Correction ####################################################
    if rcmc:
        apply_rcmc(datacube, sar_radar, r_axis, debug=debug)

        if debug:
            _plot_raw(r_axis, datacube, "After RCMC (slow-time)")

    ########## Azimuth windowing ##################################################################
    win_mat = create_window(datacube.shape, window=window, window_kwargs=window_kwargs, plot=False)
    datacube *= win_mat

    ########## Azimuth compression (focusing) #####################################################
    cross_range_axis = azimuth_matched_filter(datacube, sar_radar, r_axis)

    ########## Plot ###############################################################################
    if plot or debug:
        plot_sar_image(cross_range_axis, r_axis, datacube, "Focused SAR Image")

    return cross_range_axis, r_axis, datacube


def plot_sar_image(
    cross_range_axis: np.ndarray,
    r_axis: np.ndarray,
    data: np.ndarray,
    title: str,
    cbar_min: float = -40,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a focused SAR image in dB.

    Args:
        cross_range_axis: 1-D cross-range axis [m].
        r_axis: 1-D slant-range axis [m] (converted to km for display).
        data: 2-D complex SAR image.
        title: Plot title.
        cbar_min: Minimum colorbar value [dB].

    Returns:
        The figure and axes objects.
    """
    magnitude = np.abs(data)
    zero_to_smallest_float(magnitude)
    plot_data = 20 * np.log10(magnitude / magnitude.max())

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.set_xlabel("Cross-Range [m]")
    ax.set_ylabel("Slant Range [km]")

    mesh = ax.pcolormesh(cross_range_axis, r_axis / 1e3, plot_data)
    mesh.set_clim(cbar_min, 0)
    cbar = fig.colorbar(mesh)
    cbar.set_label("Normalised Magnitude [dB]")

    fig.tight_layout()
    return fig, ax


def _plot_raw(r_axis: np.ndarray, data: np.ndarray, title: str) -> None:
    """Plots the magnitude of a range × slow-time matrix (debug helper)."""
    pulses = range(data.shape[1])
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    mesh = ax.pcolormesh(pulses, r_axis / 1e3, np.abs(data))
    ax.set_xlabel("Pulse Index (along-track)")
    ax.set_ylabel("Slant Range [km]")
    fig.colorbar(mesh, label="Magnitude")
    fig.tight_layout()


def _plot_rdm(
    r_axis: np.ndarray,
    f_eta: np.ndarray,
    rd_data: np.ndarray,
    title: str,
) -> None:
    """Plots the magnitude of a range × azimuth-Doppler matrix (debug helper).

    Args:
        r_axis: 1-D slant-range axis [m].
        f_eta: 1-D azimuth Doppler frequency axis [Hz], in unshifted
            fft order.  ``fftshift`` is applied internally for display.
        rd_data: 2-D complex array in the range-Doppler domain
            ``(n_range_bins, n_pulses)``.
        title: Plot title.
    """
    from scipy import fft as _fft

    f_disp = _fft.fftshift(f_eta)
    mag = np.abs(_fft.fftshift(rd_data, axes=1))

    # Auto-zoom to the rows that hold most of the energy, so sub-cell
    # migration trajectories stay visible even when the full range axis
    # spans many kilometres.
    row_max = mag.max(axis=1)
    threshold = mag.max() * 10 ** (-30 / 20)
    bright_rows = np.where(row_max > threshold)[0]
    if len(bright_rows) > 0:
        k_lo = max(0, bright_rows[0] - 5)
        k_hi = min(mag.shape[0], bright_rows[-1] + 6)
    else:
        k_lo, k_hi = 0, mag.shape[0]
    mag_zoom = mag[k_lo:k_hi]
    db = 20 * np.log10(mag_zoom / mag.max() + 1e-30)

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    mesh = ax.pcolormesh(f_disp, r_axis[k_lo:k_hi] / 1e3, db, vmin=-30, vmax=0)
    ax.set_xlabel("Azimuth Doppler frequency [Hz]")
    ax.set_ylabel("Slant Range [km]")
    fig.colorbar(mesh, label="Magnitude [dB]")
    fig.tight_layout()
