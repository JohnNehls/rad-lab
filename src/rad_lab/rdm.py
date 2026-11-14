"""Range-Doppler map (RDM) generation and plotting.

Provides the `gen` entry point that simulates a full CPI — adding skin
and jammer returns to a datacube, applying matched filtering, Doppler
windowing, and the slow-time FFT — and a set of plot helpers for RTMs and RDMs.
`gen_rtm` stops before the Doppler FFT, and `noncoherent_integrate` turns its
range-time matrix into a range profile.
"""

import numpy as np
import matplotlib.pyplot as plt

from . import constants as c
from .rf_datacube import range_axis, data_cube
from .rf_datacube import matchfilter, doppler_process
from .range_equation import noise_power
from .noise import unity_variance_complex_noise
from .utilities import zero_to_smallest_float
from ._rdm_internals import add_returns, create_window, staggered_rtm
from ._rdm_extras import noise_checks
from .pulse_doppler_radar import Radar, range_rate_from_doppler
from .waveform import WaveformSample
from . import mti


def gen(
    radar: Radar,
    waveform: WaveformSample,
    return_list: list,
    seed: int = 0,
    plot: bool = True,
    debug: bool = False,
    window: str = "chebyshev",
    window_kwargs: dict | None = None,
    range_window: str = "none",
    range_window_kwargs: dict | None = None,
    mti_order: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a Range-Doppler Map (RDM) for a single Coherent Processing Interval (CPI).

    Simulates received radar data for one or more targets moving at constant
    range rates and processes it to produce an RDM, accounting for radar system
    parameters, waveform characteristics, and noise.

    Output amplitudes are in Volts (at the receiver load).  To view the RDM
    in SNR voltage ratio, pass the returned datacube through `to_snr`.
    To get a noiseless RDM for peak-finding or PSF inspection, set
    ``radar.op_temp = 0`` (thermal noise scales with temperature).

    Args:
        radar: Radar system parameters. See
            `Radar` for required keys and units.
        waveform: WaveformSample created by a factory function
            (e.g. `lfm_waveform`).
        return_list: List of `Return` objects, each
            describing one simulated target or jammer, and optionally
            `Clutter` objects, each describing a band of surface clutter.
        seed: Random number generator seed for reproducibility. Defaults to 0.
        plot: If True, plots the final RDM. Defaults to True.
        debug: If True, plots intermediate processing steps and prints
            diagnostic statistics. Defaults to False.
        window: Doppler window function applied before the slow-time FFT.
            One of ``"chebyshev"`` (default), ``"blackman-harris"``,
            ``"taylor"``, or ``"none"`` (rectangular, no windowing).
        window_kwargs: Optional dict forwarded to the underlying
            ``scipy.signal.windows`` function. For example,
            ``window_kwargs={"at": 80}`` sets Chebyshev attenuation to
            80 dB; ``window_kwargs={"nbar": 5, "sll": 35}`` tunes the
            Taylor window (``sll`` is a positive suppression in dB). See
            `create_window`.
        range_window: Weighting on the matched-filter replica for *range*
            sidelobe control (fast-time), analogous to ``window`` in Doppler.
            ``"none"`` (default) is the plain matched filter, leaving the LFM's
            ~-13.2 dB range sidelobes.  One of ``"none"``, ``"chebyshev"``,
            ``"blackman-harris"``, ``"taylor"``.  See
            `taper`.
        range_window_kwargs: Optional dict forwarded to the range-window
            function (e.g. ``{"at": 60}`` for Chebyshev).
        mti_order: Order of the MTI pulse canceller applied along slow time
            after the matched filter and before Doppler processing.  ``0``
            (default) disables MTI.  An order-``n`` canceller drops the first
            ``n`` pulses.  See `mti.apply_canceller`; pass the same value to
            `to_snr`.

    Returns:
        tuple: ``(rdot_axis, r_axis, datacube)``:

            - **rdot_axis** (*np.ndarray*): 1D range-rate (Doppler) axis [m/s].
            - **r_axis** (*np.ndarray*): 1D range axis [m].
            - **datacube** (*np.ndarray*): 2D RDM (signal + noise), amplitude in Volts.

    Raises:
        ValueError: If ``radar.pri_stagger`` is set; the Doppler FFT needs a
            constant PRI.  Use `gen_rtm` for staggered radars.
    """
    if radar.pri_stagger is not None:
        raise ValueError(
            "gen needs a constant PRI for the Doppler FFT; use gen_rtm with pri_stagger"
        )

    r_axis, datacube, noise_dc = _simulate_rtm(
        radar, waveform, return_list, seed, debug, range_window, range_window_kwargs, mti_order
    )

    ########### Doppler process ####################################################################
    chwin_norm_mat = create_window(
        datacube.shape, window=window, window_kwargs=window_kwargs, plot=False
    )
    datacube *= chwin_norm_mat

    f_axis, _ = doppler_process(datacube, radar.sample_rate)  # r_axis is unchanged

    rdot_axis = range_rate_from_doppler(f_axis, radar.fcar)

    ########## Plots and checks ####################################################################
    if debug:
        noise_checks(noise_dc, datacube)
    if plot or debug:
        plot_rdm(rdot_axis, r_axis, datacube, f"RDM for {waveform.type}")

    return rdot_axis, r_axis, datacube


def gen_rtm(
    radar: Radar,
    waveform: WaveformSample,
    return_list: list,
    seed: int = 0,
    debug: bool = False,
    range_window: str = "none",
    range_window_kwargs: dict | None = None,
    mti_order: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a range-time matrix (RTM) for a single CPI.

    The first half of `gen`: returns plus noise, matched filter, and optional
    MTI, but no Doppler window or slow-time FFT.  Use it where pulses are
    integrated non-coherently (range profiles, search displays) rather than
    coherently.  For the same arguments, applying `doppler_process` to the
    output reproduces `gen` with ``window="none"``.

    Output amplitudes are in Volts.  To get per-pulse SNR voltage ratio, pass
    the RTM through `rtm_to_snr`.

    Args:
        radar: Radar system parameters.
        waveform: WaveformSample created by a factory function.
        return_list: List of `Return` and `Clutter` objects (see `gen`).
        seed: Random number generator seed for reproducibility. Defaults to 0.
        debug: If True, plots the RTM after each processing step.
        range_window: Matched-filter replica weighting (see `gen`).
        range_window_kwargs: Optional dict forwarded to the range-window function.
        mti_order: Order of the MTI pulse canceller (see `gen`).  An
            order-``n`` canceller drops the first ``n`` pulses.

    Returns:
        tuple: ``(r_axis, rtm)``:

            - **r_axis** (*np.ndarray*): 1D range axis [m].
            - **rtm** (*np.ndarray*): 2D RTM (signal + noise), shape
              ``(n_range_bins, n_pulses - mti_order)``, amplitude in Volts.
              With ``radar.pri_stagger`` the range gate (``n_range_bins``) is
              the shortest PRI.
    """
    r_axis, rtm, _ = _simulate_rtm(
        radar, waveform, return_list, seed, debug, range_window, range_window_kwargs, mti_order
    )
    return r_axis, rtm


def _simulate_rtm(
    radar: Radar,
    waveform: WaveformSample,
    return_list: list,
    seed: int,
    debug: bool,
    range_window: str,
    range_window_kwargs: dict | None,
    mti_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Shared RTM simulation for `gen` and `gen_rtm`.

    Also returns the raw noise datacube when ``debug`` is set (for
    `noise_checks`), otherwise None.
    """
    np.random.seed(seed)

    ########## Compute waveform and radar parameters ###############################################
    waveform.set_sample(radar.sample_rate)  # set the recorded sample

    ########## Returns + noise #####################################################################
    if radar.pri_stagger is None:
        datacube = data_cube(radar.sample_rate, radar.prf, radar.n_pulses)
        add_returns(datacube, waveform, return_list, radar)
    else:
        datacube = staggered_rtm(waveform, return_list, radar)

    ########## Create range axis for plotting ######################################################
    r_axis = range_axis(radar.sample_rate, datacube.shape[0])

    # Per-sample voltage variance from the receiver noise PSD: σ² = R · N₀ · fs,
    # so a band of width B has variance R · N₀ · B as in continuous-time analysis.
    rxVolt_noise = np.sqrt(
        c.RADAR_LOAD * noise_power(radar.sample_rate, radar.noise_factor, radar.op_temp)
    )
    noise_dc = unity_variance_complex_noise(datacube.shape) * rxVolt_noise
    datacube += noise_dc

    if debug:
        plot_rtm(r_axis, datacube, "RTM: unprocessed")
    else:
        noise_dc = None  # free the cube-sized buffer before heavy processing

    ########## Match filter ########################################################################
    matchfilter(
        datacube,
        waveform.pulse_sample,
        pedantic=False,
        window=range_window,
        window_kwargs=range_window_kwargs,
    )

    if debug:
        plot_rtm(r_axis, datacube, "RTM: match filtered")

    ########## MTI #################################################################################
    if mti_order:
        datacube = mti.apply_canceller(datacube, mti_order)
        if debug:
            plot_rtm(r_axis, datacube, f"RTM: order-{mti_order} MTI")

    return r_axis, datacube, noise_dc


def _pulse_noise_voltage(radar: Radar, waveform: WaveformSample, mti_order: int) -> float:
    """Noise voltage std of one matched-filtered (and MTI'd) pulse sample.

    With a unit-amplitude template of ``N_taps = T·fs`` samples, the raw MF
    output noise variance is ``N_taps · R·N₀·fs = R·N₀·T·fs²``, and the
    canceller multiplies it by `mti.noise_gain`.
    """
    # noise_power(B=1, F, T) = k·T·F is the one-sided PSD N₀ [W/Hz].
    n0_R = c.RADAR_LOAD * noise_power(1.0, radar.noise_factor, radar.op_temp)
    return radar.sample_rate * np.sqrt(mti.noise_gain(mti_order) * n0_R * waveform.pulse_width)


def to_snr(
    datacube: np.ndarray, radar: Radar, waveform: WaveformSample, mti_order: int = 0
) -> np.ndarray:
    """Convert a Volt-domain RDM to SNR voltage ratio.

    Normalises so the peak magnitude equals the range-equation SNR voltage
    and the off-peak noise floor has voltage standard deviation of 1.  Use
    `verify_snr` to verify against theory.  With MTI the noise is no longer
    white: its floor follows the canceller's ``|H(f)|^2`` across Doppler, so
    the unit standard deviation holds only on average over Doppler bins.

    Derivation, with unit-amplitude pulse template (``|p|=1`` over duration
    ``T``, ``N_taps = T·fs`` samples) and raw (unscaled) matched filter:

    - Per ADC sample: noise variance ``σ² = R·N₀·fs``.
    - After MF (raw sum over ``N_taps`` unit-amplitude taps):
      noise variance = ``N_taps · σ² = R·N₀·T·fs²``.
    - After coherent integration of ``N`` pulses by the slow-time FFT:
      noise variance = ``N · R·N₀·T·fs²``.

    So dividing by ``fs · sqrt(N · R·N₀·T)`` yields the SNR voltage ratio.

    An order-``n`` MTI canceller leaves ``N - n`` pulses and multiplies the
    white-noise variance by `mti.noise_gain`, so both enter the normalisation.

    Args:
        datacube: Processed RDM returned by `gen`, in Volts.
        radar: Same radar used to generate the RDM.
        waveform: Same waveform used to generate the RDM.
        mti_order: Same ``mti_order`` passed to `gen`.  Defaults to 0.

    Returns:
        np.ndarray: RDM normalised to SNR voltage ratio.
    """
    n_pulses_out = radar.n_pulses - mti_order
    noise_v_out = np.sqrt(n_pulses_out) * _pulse_noise_voltage(radar, waveform, mti_order)
    return datacube / noise_v_out


def rtm_to_snr(
    rtm: np.ndarray, radar: Radar, waveform: WaveformSample, mti_order: int = 0
) -> np.ndarray:
    """Convert a Volt-domain RTM to per-pulse SNR voltage ratio.

    The single-pulse counterpart of `to_snr`: divides by the noise voltage of
    one matched-filtered pulse, so each RTM sample has unit noise variance and
    a target's peak ``|x|^2`` equals its single-pulse SNR.

    Args:
        rtm: RTM returned by `gen_rtm`, in Volts.
        radar: Same radar used to generate the RTM.
        waveform: Same waveform used to generate the RTM.
        mti_order: Same ``mti_order`` passed to `gen_rtm`.  Defaults to 0.

    Returns:
        np.ndarray: RTM normalised to per-pulse SNR voltage ratio.
    """
    return rtm / _pulse_noise_voltage(radar, waveform, mti_order)


def noncoherent_integrate(rtm: np.ndarray) -> np.ndarray:
    """Range profile by non-coherent integration: ``sum_n |x[n]|^2`` over pulses.

    The sum discards phase, so it needs no Doppler knowledge.  For an RTM from
    `rtm_to_snr`, the noise-only output is a sum of ``N`` unit-mean
    exponentials (chi-squared with ``2N`` DOF, halved), which is the statistic
    `detection.threshold_factor_nci` thresholds: compare the profile directly
    to ``threshold_factor_nci(pfa, N)``.

    Args:
        rtm: 2D RTM, shape ``(n_range_bins, n_pulses)``.

    Returns:
        np.ndarray: 1D integrated power profile, shape ``(n_range_bins,)``.
    """
    return np.sum(np.abs(rtm) ** 2, axis=1)


def doppler_bank(rtm: np.ndarray, pulse_times: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Coherent Doppler filter bank for arbitrarily spaced pulses.

    Computes ``y[r, f] = sum_n x[r, n] exp(-j 2 pi f t_n)``, the discrete-time
    Fourier transform evaluated at the actual pulse times.  For evenly spaced
    pulses and ``freqs`` on the FFT grid this equals the (unshifted) slow-time
    FFT of `doppler_process`; with a staggered PRI it is the direct
    replacement.  Same sign convention: a target at range rate ``v`` peaks at
    ``f = -2 v fcar / c``.

    A target at a bin's Doppler adds in phase, so the coherent gain is the
    full ``N`` whatever the spacing, while unit-variance white noise has
    variance ``N`` (``sum |w|^2`` if the RTM was tapered by weights ``w``).

    Args:
        rtm: 2D RTM, shape ``(n_range_bins, n_pulses)``.
        pulse_times: Time of each RTM column [s], e.g. ``radar.pulse_tx_times``
            (``[mti_order:]`` after an MTI canceller).
        freqs: Doppler frequencies of the filters [Hz].

    Returns:
        np.ndarray: Complex range-Doppler map, shape ``(n_range_bins, len(freqs))``.
    """
    steering = np.exp(-2j * np.pi * pulse_times[:, None] * freqs[None, :])
    return rtm @ steering


def plot_rtm(r_axis: np.ndarray, data: np.ndarray, title: str) -> None:
    """Plots the magnitude and phase of a range-time matrix (RTM).

    The RTM shows radar data before Doppler processing, with range on one
    axis and pulse number (slow-time) on the other.

    Args:
        r_axis: 1D array of range values in meters.
        data: 2D complex array representing the RTM, with shape
              (num_range_bins, num_pulses).
        title: The title for the plot.
    """
    pulses = range(data.shape[1])
    fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    mag_plot = ax_mag.pcolormesh(pulses, r_axis * 1e-3, np.abs(data))
    ax_mag.set_xlabel("Pulse Number")
    ax_mag.set_ylabel("Range [km]")
    ax_mag.set_title("Magnitude")
    fig.colorbar(mag_plot, ax=ax_mag)

    phase_plot = ax_phase.pcolormesh(pulses, r_axis * 1e-3, np.angle(data))
    ax_phase.set_xlabel("Pulse Number")
    ax_phase.set_ylabel("Range [km]")
    ax_phase.set_title("Phase")
    fig.colorbar(phase_plot, ax=ax_phase)

    fig.tight_layout()
    plt.show()


def plot_rdm(
    rdot_axis: np.ndarray,
    r_axis: np.ndarray,
    data: np.ndarray,
    title: str,
    cbar_min: float = -100,
    volt_to_dbm: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a range-Doppler matrix (RDM).

    The RDM shows radar data after pulse compression and Doppler processing.

    Args:
        rdot_axis: 1D array of range-rate values in m/s.
        r_axis: 1D array of range values in meters.
        data: 2D complex array representing the RDM.
        title: The title for the plot.
        cbar_min: The minimum value for the color bar. Defaults to -100.
        volt_to_dbm: If True, converts data from voltage to dBm for plotting.
                       If False, plots power in Watts. Defaults to True.

    Returns:
        The figure and axes objects of the plot.
    """
    magnitude_data = np.abs(data)

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.set_xlabel("Range Rate [km/s]")
    ax.set_ylabel("Range [km]")

    if volt_to_dbm:
        zero_to_smallest_float(magnitude_data)
        # P_dBm = 10*log10(P_W / 1mW) = 10*log10((V^2/R) / 1e-3)
        plot_data = 20 * np.log10(magnitude_data / np.sqrt(1e-3 * c.RADAR_LOAD))
        cbar_label = "Power [dBm]"
    else:
        # P_W = V^2 / R
        plot_data = magnitude_data**2 / c.RADAR_LOAD
        cbar_label = "Power [W]"

    mesh = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, plot_data)
    mesh.set_clim(cbar_min, plot_data.max())
    cbar = fig.colorbar(mesh)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    return fig, ax


def plot_rdm_snr(
    rdot_axis: np.ndarray,
    r_axis: np.ndarray,
    data: np.ndarray,
    title: str,
    cbar_min: float = 0,
    volt_ratio_to_db: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a range-Doppler matrix in terms of Signal-to-Noise Ratio (SNR).

    Args:
        rdot_axis: 1D array of range-rate values in m/s.
        r_axis: 1D array of range values in meters.
        data: 2D array representing the RDM with amplitudes as a linear SNR
              voltage ratio (i.e., S_voltage / N_voltage).
        title: The title for the plot.
        cbar_min: The minimum value for the color bar. Defaults to 0.
        volt_ratio_to_db: If True, converts the SNR voltage ratio to dB.
                            Defaults to True.

    Returns:
        The figure and axes objects of the plot.
    """
    snr_voltage_ratio = np.abs(data)
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.set_xlabel("Range Rate [km/s]")
    ax.set_ylabel("Range [km]")

    if volt_ratio_to_db:
        zero_to_smallest_float(snr_voltage_ratio)
        plot_data = 20 * np.log10(snr_voltage_ratio)
        cbar_label = "SNR [dB]"
    else:
        plot_data = snr_voltage_ratio
        cbar_label = "SNR (Voltage Ratio)"

    mesh = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, plot_data)
    mesh.set_clim(cbar_min, plot_data.max())
    cbar = fig.colorbar(mesh)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    return fig, ax
