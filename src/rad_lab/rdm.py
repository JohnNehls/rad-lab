"""Range-Doppler map (RDM) generation and plotting.

Provides the :func:`gen` entry point that simulates a full CPI — adding skin
and jammer returns to a datacube, applying matched filtering, Doppler
windowing, and the slow-time FFT — and a set of plot helpers for RTMs and RDMs.
"""

import numpy as np
import matplotlib.pyplot as plt

from . import constants as c
from .rf_datacube import number_range_bins, range_axis, data_cube
from .rf_datacube import matchfilter, doppler_process
from .range_equation import noise_power
from .noise import unity_variance_complex_noise
from .utilities import zero_to_smallest_float
from ._rdm_internals import add_returns, create_window
from ._rdm_extras import noise_checks
from .pulse_doppler_radar import Radar
from .waveform import WaveformSample


def gen(
    radar: Radar,
    waveform: WaveformSample,
    return_list: list,
    seed: int = 0,
    plot: bool = True,
    debug: bool = False,
    window: str = "chebyshev",
    window_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a Range-Doppler Map (RDM) for a single Coherent Processing Interval (CPI).

    Simulates received radar data for one or more targets moving at constant
    range rates and processes it to produce an RDM, accounting for radar system
    parameters, waveform characteristics, and noise.

    Output amplitudes are in Volts (at the receiver load).  To view the RDM
    in SNR voltage ratio, pass the returned datacube through :func:`to_snr`.
    To get a noiseless RDM for peak-finding or PSF inspection, set
    ``radar.op_temp = 0`` (thermal noise scales with temperature).

    Args:
        radar: Radar system parameters. See
            :class:`rad_lab.pulse_doppler_radar.Radar` for required keys and units.
        waveform: WaveformSample created by a factory function
            (e.g. :func:`rad_lab.waveform.lfm_waveform`).
        return_list: List of :class:`rad_lab.returns.Return` objects, each
            describing one simulated target or jammer.
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
            80 dB; ``window_kwargs={"nbar": 5, "sll": -35}`` tunes the
            Taylor window. See :func:`._rdm_internals.create_window`.

    Returns:
        tuple: ``(rdot_axis, r_axis, datacube)``:

            - **rdot_axis** (*np.ndarray*): 1D range-rate (Doppler) axis [m/s].
            - **r_axis** (*np.ndarray*): 1D range axis [m].
            - **datacube** (*np.ndarray*): 2D RDM (signal + noise), amplitude in Volts.
    """
    np.random.seed(seed)

    ########## Compute waveform and radar parameters ###############################################
    waveform.set_sample(radar.sample_rate)  # set the recorded sample

    ########## Create range axis for plotting ######################################################
    r_axis = range_axis(radar.sample_rate, number_range_bins(radar.sample_rate, radar.prf))

    ########## Returns + noise #####################################################################
    datacube = data_cube(radar.sample_rate, radar.prf, radar.n_pulses)
    add_returns(datacube, waveform, return_list, radar)

    rxVolt_noise = np.sqrt(
        c.RADAR_LOAD * noise_power(waveform.bw, radar.noise_factor, radar.op_temp)
    )
    noise_dc = unity_variance_complex_noise(datacube.shape) * rxVolt_noise
    datacube += noise_dc

    if debug:
        plot_rtm(r_axis, datacube, "RTM: unprocessed")
    else:
        del noise_dc  # free the cube-sized buffer before heavy processing

    ########## Match filter ########################################################################
    matchfilter(datacube, waveform.pulse_sample, pedantic=False)

    if debug:
        plot_rtm(r_axis, datacube, "RTM: match filtered")

    ########### Doppler process ####################################################################
    chwin_norm_mat = create_window(
        datacube.shape, window=window, window_kwargs=window_kwargs, plot=False
    )
    datacube *= chwin_norm_mat

    f_axis, r_axis = doppler_process(datacube, radar.sample_rate)

    # f = -2 fc/c * Rdot -> Rdot = -c f / (2 fc)
    rdot_axis = -c.C * f_axis / (2 * radar.fcar)

    ########## Plots and checks ####################################################################
    if debug:
        noise_checks(noise_dc, datacube)
    if plot or debug:
        plot_rdm(rdot_axis, r_axis, datacube, f"RDM for {waveform.type}")

    return rdot_axis, r_axis, datacube


def to_snr(datacube: np.ndarray, radar: Radar, waveform: WaveformSample) -> np.ndarray:
    """Convert a Volt-domain RDM to SNR voltage ratio.

    Normalises so the peak magnitude equals the range-equation SNR and
    the off-peak noise floor has voltage standard deviation of 1.  Use
    :func:`~rad_lab._rdm_extras.verify_snr` to verify against theory.

    Args:
        datacube: Processed RDM returned by :func:`gen`, in Volts.
        radar: Same radar used to generate the RDM.
        waveform: Same waveform used to generate the RDM.

    Returns:
        np.ndarray: RDM normalised to SNR voltage ratio.
    """
    # Matched filter delivers variance gain TB (pulse is scaled to sum|p|^2 = TB);
    # slow-time FFT adds a further factor of N in variance.  So the output noise
    # voltage std is rx_v_in * sqrt(N * TB).
    noise_v_in = np.sqrt(
        c.RADAR_LOAD * noise_power(waveform.bw, radar.noise_factor, radar.op_temp)
    )
    noise_v_out = noise_v_in * np.sqrt(radar.n_pulses * waveform.time_bw_product)
    return datacube / noise_v_out


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
