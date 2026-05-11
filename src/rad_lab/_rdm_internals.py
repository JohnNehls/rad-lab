import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from scipy import signal
from . import constants as c
from .waveform_helpers import add_waveform_at_index
from .utilities import phase_negpi_pospi
from .range_equation import snr_range_eqn, signal_range_eqn, signal_range_eqn_one_way
from . import vbm
from .pulse_doppler_radar import Radar
from .waveform import WaveformSample
from .returns import Target, EaPlatform, Return


def _propagation_phase(delays: np.ndarray, fcar: float) -> np.ndarray:
    """Returns the carrier phase accumulated over one-way or two-way propagation delays."""
    return -2 * np.pi * fcar * delays


def _return_sample_indices(
    return_times: np.ndarray, waveform: WaveformSample, sample_rate: float
) -> np.ndarray:
    """Converts pulse return times to flat datacube sample indices.

    Subtracts half the pulse width since pulses are timed from their leading edge.
    The range axis is 1-indexed — r_axis[k] = (k+1)*dR — so the injection index
    must be one less than round(t*fs) to land the matched filter peak in the
    correct bin.
    """
    times_of_arrival = return_times - waveform.pulse_width / 2
    return np.round(times_of_arrival * sample_rate).astype(int) - 1


@contextmanager
def _flat_datacube(datacube: np.ndarray):
    """Yields a flattened view of the datacube and writes it back on exit.

    The datacube's slow-time axis is non-contiguous, so it must be transposed
    before flattening to produce a contiguous pulse-major layout.
    """
    flat = datacube.T.flatten()
    yield flat
    datacube[:] = flat.reshape(tuple(reversed(datacube.shape))).T


def _inject_pulses(
    datacube: np.ndarray,
    waveform_samples: np.ndarray,
    sample_indices: np.ndarray,
    phases: np.ndarray,
    amplitude: complex | float | np.ndarray,
) -> None:
    """Adds phase-shifted, scaled copies of a waveform to the datacube.

    For each pulse, constructs ``amplitude * waveform_samples * exp(j * phases[i])``
    and adds it at the corresponding flat-datacube index.  Pulses whose index
    exceeds the datacube size are silently skipped.

    Args:
        datacube: 2-D complex array modified in place.
        waveform_samples: Discrete pulse samples from the waveform.
        sample_indices: Per-pulse starting index in the flattened datacube.
        phases: Per-pulse carrier phase [rad].
        amplitude: Amplitude scaling.  A scalar applies the same value to
            every pulse; a 1-D array of length ``n_pulses`` provides
            per-pulse weighting (e.g. beam-pattern gains in spotlight SAR).
    """
    with _flat_datacube(datacube) as flat:
        for i in range(len(sample_indices)):
            if sample_indices[i] < datacube.size:
                amp_i = amplitude[i] if np.ndim(amplitude) > 0 else amplitude
                pulse = amp_i * waveform_samples * np.exp(1j * phases[i])
                add_waveform_at_index(flat, pulse, sample_indices[i])


def add_skin(
    datacube: np.ndarray,
    waveform: WaveformSample,
    tgt_info: Target,
    radar: Radar,
    return_magnitude: float,
) -> None:
    """Adds a direct radar reflection (skin return) from a target to the datacube.

    This function simulates the signal received by the radar after it reflects
    off a target. The function calculates the time delay and phase shift for each
    pulse and adds the appropriately modified waveform to the datacube.
    The datacube is modified in place.

    Uses the stop-and-hop assumption: target range is evaluated once per pulse,
    at the pulse-transmit instant (`pulse_tx_times`), and that single value
    drives both the round-trip delay and the carrier phase. The target is
    treated as stationary during the pulse and only "hops" between PRIs.

    Args:
        datacube: 2D complex array to which the return is added.
        waveform: WaveformSample containing pulse data and parameters.
        tgt_info: Target kinematics and scattering parameters.
        radar: Radar system parameters.
        return_magnitude: The voltage or SNR amplitude of the return for a single pulse.
    """
    pulse_tx_times = np.arange(radar.n_pulses) / radar.prf
    target_range_per_pulse = tgt_info.range + tgt_info.range_rate * pulse_tx_times
    two_way_delays = 2 * target_range_per_pulse / c.C
    pulse_return_times = pulse_tx_times + two_way_delays
    two_way_doppler_phases = _propagation_phase(two_way_delays, radar.fcar)
    return_sample_indices = _return_sample_indices(pulse_return_times, waveform, radar.sample_rate)

    _inject_pulses(
        datacube,
        waveform.pulse_sample,
        return_sample_indices,
        two_way_doppler_phases,
        amplitude=return_magnitude * tgt_info.sv,
    )


def add_jammer(
    datacube: np.ndarray,
    waveform: WaveformSample,
    radar: Radar,
    return_info: Return,
    return_magnitude: float,
) -> None:
    """Adds a DRFM jammer return to the datacube.

    This function simulates a DRFM jammer that records an incoming pulse and
    re-transmits it with modifications to deceive the radar.
    The datacube is modified in place.

    Args:
        datacube: 2D complex array to which the return is added.
        waveform: WaveformSample containing pulse data and parameters.
        radar: Radar system parameters.
        return_info: Return describing the EA platform and target parameters.
        return_magnitude: The voltage or SNR amplitude of the return.
    """
    target = return_info.target
    pulse_tx_times = np.arange(radar.n_pulses) / radar.prf

    # Calculate timing and phase for the signal's one-way trip to the target
    target_range_per_pulse = target.range + target.range_rate * pulse_tx_times
    one_way_delays = target_range_per_pulse / c.C
    skin_return_times = pulse_tx_times + 2 * one_way_delays
    one_way_propagation_phases = _propagation_phase(one_way_delays, radar.fcar)

    ea = return_info.platform

    # Doppler frequency shift for range-rate offset
    doppler_freq_offset = 2 * radar.fcar / c.C * ea.rdot_offset

    # Phase modulation for Velocity Bin Masking (VBM)
    if ea.rdot_delta is not None:
        vbm_noise_function = ea.vbm_noise_function or vbm._lfm_phase
        slowtime_noise = vbm.slowtime_noise(
            radar.n_pulses,
            radar.fcar,
            ea.rdot_delta,
            radar.prf,
            noise_fun=vbm_noise_function,
        )
    else:
        slowtime_noise = np.ones(radar.n_pulses)

    # Additional time delay for range offset
    total_delay = ea.delay + 2 * ea.range_offset / c.C
    return_times = skin_return_times + total_delay
    return_sample_indices = _return_sample_indices(return_times, waveform, radar.sample_rate)

    # Precompute per-pulse rdot-offset phase shift vector
    pulse_indices = np.arange(radar.n_pulses)
    rdot_phase = np.exp(-1j * pulse_indices * 2 * np.pi * doppler_freq_offset / radar.prf)

    stored_pulse = 0
    stored_angle = 0

    with _flat_datacube(datacube) as flat:
        for i in range(radar.n_pulses):
            received_pulse = waveform.pulse_sample * np.exp(1j * one_way_propagation_phases[i])

            if i == 0:
                stored_pulse = received_pulse
                continue

            if i == 1:
                # Estimate target's Doppler phase shift between pulses
                angle_diff = np.angle(received_pulse) - np.angle(stored_pulse)
                stored_angle = np.mean(phase_negpi_pospi(angle_diff))

            pulse = (
                return_magnitude
                * stored_pulse
                * target.sv
                * slowtime_noise[i]
                * np.exp(1j * i * stored_angle)
                * rdot_phase[i]
                * np.exp(1j * one_way_propagation_phases[i])
            )

            if return_sample_indices[i] < datacube.size:
                add_waveform_at_index(flat, pulse, return_sample_indices[i])


def create_window(
    shape: tuple[int, int],
    window: str = "chebyshev",
    window_kwargs: dict | None = None,
    plot: bool = False,
) -> np.ndarray:
    """Creates a 2D window matrix for Doppler sidelobe reduction.

    The window is applied along the slow-time (pulse) dimension before the
    Doppler FFT. Windowing trades sidelobe suppression for mainlobe width.

    Args:
        shape: Desired shape of the output window (num_range_bins, num_pulses).
        window: Window type. One of:
            - ``"chebyshev"`` (default): equi-ripple sidelobes. Accepts
              ``window_kwargs={"at": <dB>}`` (default 60 dB).
            - ``"blackman-harris"``: very low sidelobes (~-92 dB), wider
              mainlobe than Chebyshev.
            - ``"taylor"``: low near-in sidelobes with a good compromise
              between resolution and sidelobe level. Accepts
              ``window_kwargs={"nbar": ..., "sll": ..., "norm": ...}``.
            - ``"none"``: rectangular (no windowing), narrowest mainlobe
              but highest sidelobes (~-13 dB).
        window_kwargs: Optional dict of keyword arguments forwarded to the
            underlying ``scipy.signal.windows`` function. Keys must match
            the scipy function's parameter names (e.g. ``"at"`` for
            Chebyshev, ``"nbar"``/``"sll"`` for Taylor). Ignored for
            ``"blackman-harris"`` and ``"none"``.
        plot: If True, displays a plot of the generated window.

    Returns:
        The 2D window matrix of shape ``shape``, normalized so that its mean
        is 1.0 (preserving coherent gain).
    """
    assert len(shape) == 2, "Shape must be a 2-element tuple."
    num_range_bins, num_pulses = shape
    kwargs = window_kwargs or {}

    window_lower = window.lower()
    if window_lower == "chebyshev":
        at = kwargs.get("at", 60.0)
        win_1d = signal.windows.chebwin(num_pulses, at)
    elif window_lower == "blackman-harris":
        win_1d = signal.windows.blackmanharris(num_pulses)
    elif window_lower == "taylor":
        win_1d = signal.windows.taylor(num_pulses, **kwargs)
    elif window_lower == "none":
        win_1d = np.ones(num_pulses)
    else:
        raise ValueError(
            f"Unknown window type '{window}'. "
            "Choose from: 'chebyshev', 'blackman-harris', 'taylor', 'none'."
        )

    normalized_win = win_1d / np.mean(win_1d)
    window_matrix = np.tile(normalized_win, (num_range_bins, 1))

    if plot:
        plt.figure()
        plt.title(f"{window} Window")
        plt.imshow(window_matrix)
        plt.xlabel("Slow Time (Pulses)")
        plt.ylabel("Fast Time (Range Bins)")
        plt.colorbar(label="Amplitude")
        plt.show()

    return window_matrix


def skin_snr_amplitude(radar: Radar, target: Target, waveform: WaveformSample) -> float:
    """Calculates the required per-pulse voltage amplitude to achieve a target SNR.

    Uses the radar range equation to find the SNR after processing, then works
    backward to determine the necessary per-pulse signal amplitude to inject
    into the simulation datacube.

    Args:
        radar: Radar system parameters.
        target: Target kinematics and scattering parameters.
        waveform: WaveformSample containing pulse data and parameters.

    Returns:
        The required per-pulse SNR as a linear voltage ratio.
    """
    # Assumes the range equation provides the total SNR after coherent integration
    # over all pulses in the Coherent Processing Interval (CPI).
    snr_after_integration = snr_range_eqn(
        radar.tx_power,
        radar.tx_gain,
        radar.rx_gain,
        target.rcs,
        c.C / radar.fcar,
        target.range,
        waveform.bw,
        radar.noise_factor,
        radar.total_losses,
        radar.op_temp,
        waveform.time_bw_product,
    )

    # To find the required per-pulse amplitude, we first find the per-pulse SNR
    # by dividing by the number of pulses (the coherent integration gain).
    snr_per_pulse = snr_after_integration / radar.n_pulses

    # The voltage amplitude for a single pulse is the square root of the per-pulse
    # SNR (power ratio), assuming a normalized noise power of 1.0.
    return np.sqrt(snr_per_pulse)


def skin_voltage_amplitude(radar: Radar, target: Target) -> float:
    """Calculates the received voltage amplitude of a skin return.

    Args:
        radar: Radar system parameters.
        target: Target kinematics and scattering parameters.

    Returns:
        The received voltage amplitude.
    """
    return np.sqrt(
        c.RADAR_LOAD
        * signal_range_eqn(
            radar.tx_power,
            radar.tx_gain,
            radar.rx_gain,
            target.rcs,
            c.C / radar.fcar,
            target.range,
            radar.total_losses,
        )
    )


def jammer_voltage_amplitude(platform: EaPlatform, radar: Radar, target: Target) -> float:
    """Calculates the received voltage amplitude of a DRFM jammer return.

    Models the one-way communication link from the EA platform to the radar
    using the Friis equation.

    Args:
        platform: EA platform transmitter parameters.
        radar: Radar system parameters (as receiver).
        target: Target kinematics (range is used for path loss).

    Returns:
        The received voltage amplitude from the EA platform.
    """
    return np.sqrt(
        c.RADAR_LOAD
        * signal_range_eqn_one_way(
            platform.tx_power,
            platform.tx_gain,
            radar.rx_gain,
            c.C / radar.fcar,
            target.range,
            platform.total_losses,
        )
    )


def add_returns(
    datacube: np.ndarray,
    waveform: WaveformSample,
    return_list: list,
    radar: Radar,
) -> None:
    """Adds multiple returns to a datacube.

    For each Return in return_list:
    - A skin return is added when ``target.rcs is not None``.
    - A jammer return is added when ``platform is not None``.
    Both can fire for the same Return, modelling a co-located jammer on the target.

    Amplitudes are physically-based voltages at the receiver load.  The
    datacube is modified in place.

    Args:
        datacube: The 2D complex datacube to modify.
        waveform: WaveformSample containing pulse data and parameters.
        return_list: A list of Return objects.
        radar: Radar system parameters.
    """
    for item in return_list:
        if not isinstance(item, Return):
            print(f"Return type '{type(item).__name__}' not recognized. No return added.")
            continue

        if item.target.rcs is not None:
            amp = skin_voltage_amplitude(radar, item.target)
            add_skin(datacube, waveform, item.target, radar, amp)

        if item.platform is not None:
            amp = jammer_voltage_amplitude(item.platform, radar, item.target)
            add_jammer(datacube, waveform, radar, item, amp)
