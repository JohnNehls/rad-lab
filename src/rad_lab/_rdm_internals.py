import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from . import constants as c
from .waveform_helpers import add_waveform_at_index
from .rf_datacube import taper
from .utilities import phase_negpi_pospi
from .range_equation import radar_signal_voltage, signal_range_eqn_one_way
from .noise import unity_variance_complex_noise
from . import vbm
from .pulse_doppler_radar import Radar, frequency_delta_doppler
from .waveform import WaveformSample
from .returns import Target, EaPlatform, Return, Clutter


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
    falls outside the datacube (before sample 0 or beyond the end) are
    silently skipped — a negative index would otherwise wrap around and
    corrupt the end of the buffer.

    Args:
        datacube: 2-D complex array modified in place.
        waveform_samples: Discrete pulse samples from the waveform.
        sample_indices: Per-pulse starting index in the flattened datacube.
        phases: Per-pulse carrier phase [rad].
        amplitude: Amplitude scaling.  A scalar applies the same value to
            every pulse; a 1-D array of length ``n_pulses`` provides
            per-pulse weighting (e.g. beam-pattern gains in spotlight SAR).
    """
    amplitudes = np.broadcast_to(amplitude, len(sample_indices))
    with _flat_datacube(datacube) as flat:
        for i in range(len(sample_indices)):
            if 0 <= sample_indices[i] < datacube.size:
                pulse = amplitudes[i] * waveform_samples * np.exp(1j * phases[i])
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
    pulse_tx_times = radar.pulse_tx_times
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
    if radar.pri_stagger is not None:
        raise NotImplementedError("jammer returns assume a constant PRI (pri_stagger=None)")

    target = return_info.target
    pulse_tx_times = radar.pulse_tx_times

    # Calculate timing and phase for the signal's one-way trip to the target
    target_range_per_pulse = target.range + target.range_rate * pulse_tx_times
    one_way_delays = target_range_per_pulse / c.C
    skin_return_times = pulse_tx_times + 2 * one_way_delays
    one_way_propagation_phases = _propagation_phase(one_way_delays, radar.fcar)

    ea = return_info.platform

    # Doppler frequency shift for range-rate offset
    doppler_freq_offset = frequency_delta_doppler(ea.rdot_offset, radar.fcar)

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
    rdot_phase = np.exp(1j * pulse_indices * 2 * np.pi * doppler_freq_offset / radar.prf)

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

            if 0 <= return_sample_indices[i] < datacube.size:
                add_waveform_at_index(flat, pulse, return_sample_indices[i])


def clutter_cell_rcs(clutter: Clutter, range: float, range_spacing: float) -> float:
    """Constant-gamma RCS of one clutter cell [m^2].

    ``sigma = gamma * sin(psi) * A`` with illuminated area
    ``A = R * theta_az * dR / cos(psi)``.

    Args:
        clutter: Clutter band parameters.
        range: Range to the cell [m].
        range_spacing: Range extent of the cell [m] (the datacube sample spacing).
    """
    psi = clutter.grazing_angle
    area = range * clutter.azimuth_beamwidth * range_spacing / np.cos(psi)
    return clutter.gamma * np.sin(psi) * area


def gaussian_spectrum_sequence(n_pulses: int, prf: float, doppler_std: float) -> np.ndarray:
    """Unit-power complex Gaussian slow-time sequence with a Gaussian Doppler spectrum.

    White complex noise is shaped in the frequency domain by the square root of
    a zero-mean Gaussian PSD with standard deviation ``doppler_std``, then
    normalised so the expected power per pulse is 1.  ``doppler_std = 0`` gives
    a perfectly stationary cell: one random complex amplitude for every pulse.

    Args:
        n_pulses: Sequence length.
        prf: Pulse repetition frequency [Hz].
        doppler_std: Standard deviation of the Doppler spectrum [Hz].
    """
    if doppler_std == 0:
        return np.full(n_pulses, unity_variance_complex_noise(1)[0])

    freqs = np.fft.fftfreq(n_pulses, d=1 / prf)
    amplitude_response = np.exp(-(freqs**2) / (4 * doppler_std**2))  # sqrt of Gaussian PSD
    amplitude_response /= np.sqrt(np.mean(amplitude_response**2))  # unit expected power
    white = unity_variance_complex_noise(n_pulses)
    return np.fft.ifft(np.fft.fft(white) * amplitude_response)


def gaussian_spectrum_factor(times: np.ndarray, doppler_std: float) -> np.ndarray:
    """Factor ``L`` with ``L @ L^H = R`` for a Gaussian-spectrum process sampled at ``times``.

    The counterpart of `gaussian_spectrum_sequence` for unevenly spaced pulses
    (staggered PRI), where the FFT shaping does not apply.  A zero-mean
    Gaussian PSD of standard deviation ``sigma_f`` has autocorrelation
    ``R(tau) = exp(-2 pi^2 sigma_f^2 tau^2)``, so ``L @ w`` for unit-variance
    white ``w`` is a unit-power sequence with that spectrum.  ``L`` comes from
    an eigendecomposition rather than Cholesky because ``R`` is nearly
    singular for narrow spectra (rank 1 for ``doppler_std = 0``).

    Args:
        times: Pulse times [s].
        doppler_std: Standard deviation of the Doppler spectrum [Hz].
    """
    tau = times[:, None] - times[None, :]
    covariance = np.exp(-2 * np.pi**2 * doppler_std**2 * tau**2)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    return eigvecs * np.sqrt(np.clip(eigvals, 0, None))


def add_clutter(
    datacube: np.ndarray,
    waveform: WaveformSample,
    clutter: Clutter,
    radar: Radar,
) -> None:
    """Adds distributed surface clutter to the datacube.

    The clutter band is split into one cell per range sample.  Each cell is
    injected like a skin return moving at ``clutter.range_rate_mean`` (same
    stop-and-hop delay and phase as `add_skin`), but its per-pulse amplitude is
    a random sequence from `gaussian_spectrum_sequence` scaled by the
    range-equation voltage of the cell's constant-gamma RCS.  The random
    sequence spreads the cell's energy over the clutter Doppler spectrum.
    With a staggered PRI the sequence comes from `gaussian_spectrum_factor`
    at the actual pulse times instead.

    Args:
        datacube: 2D complex array to which the clutter is added.
        waveform: WaveformSample containing pulse data and parameters.
        clutter: Clutter band parameters.
        radar: Radar system parameters.
    """
    range_spacing = c.C / (2 * radar.sample_rate)
    first_cell = int(np.ceil(clutter.range_start / range_spacing))
    last_cell = int(np.floor(clutter.range_stop / range_spacing))
    cell_ranges = np.arange(first_cell, last_cell + 1) * range_spacing

    wavelength = c.C / radar.fcar
    doppler_std = 2 * clutter.range_rate_std / wavelength
    pulse_tx_times = radar.pulse_tx_times
    if radar.pri_stagger is not None:
        stagger_factor = gaussian_spectrum_factor(pulse_tx_times, doppler_std)

    for cell_range in cell_ranges:
        cell_rcs = clutter_cell_rcs(clutter, cell_range, range_spacing)
        cell_voltage = radar_signal_voltage(radar, cell_rcs, cell_range)

        if radar.pri_stagger is None:
            slow_time = gaussian_spectrum_sequence(radar.n_pulses, radar.prf, doppler_std)
        else:
            slow_time = stagger_factor @ unity_variance_complex_noise(radar.n_pulses)

        cell_range_per_pulse = cell_range + clutter.range_rate_mean * pulse_tx_times
        two_way_delays = 2 * cell_range_per_pulse / c.C
        pulse_return_times = pulse_tx_times + two_way_delays
        _inject_pulses(
            datacube,
            waveform.pulse_sample,
            _return_sample_indices(pulse_return_times, waveform, radar.sample_rate),
            _propagation_phase(two_way_delays, radar.fcar),
            amplitude=cell_voltage * slow_time,
        )


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
        is 1.0 (preserving coherent gain).  Each row is `taper`.
    """
    assert len(shape) == 2, "Shape must be a 2-element tuple."
    num_range_bins, num_pulses = shape
    window_matrix = np.tile(taper(num_pulses, window, window_kwargs), (num_range_bins, 1))

    if plot:
        plt.figure()
        plt.title(f"{window} Window")
        plt.imshow(window_matrix)
        plt.xlabel("Slow Time (Pulses)")
        plt.ylabel("Fast Time (Range Bins)")
        plt.colorbar(label="Amplitude")
        plt.show()

    return window_matrix


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
    Each Clutter in return_list adds a band of distributed clutter.

    Amplitudes are physically-based voltages at the receiver load.  The
    datacube is modified in place.

    Args:
        datacube: The 2D complex datacube to modify.
        waveform: WaveformSample containing pulse data and parameters.
        return_list: A list of Return and Clutter objects.
        radar: Radar system parameters.
    """
    for item in return_list:
        if isinstance(item, Clutter):
            add_clutter(datacube, waveform, item, radar)
            continue

        if not isinstance(item, Return):
            print(f"Return type '{type(item).__name__}' not recognized. No return added.")
            continue

        if item.target.rcs is not None:
            amp = radar_signal_voltage(radar, item.target.rcs, item.target.range)
            add_skin(datacube, waveform, item.target, radar, amp)

        if item.platform is not None:
            amp = jammer_voltage_amplitude(item.platform, radar, item.target)
            add_jammer(datacube, waveform, radar, item, amp)


def staggered_rtm(waveform: WaveformSample, return_list: list, radar: Radar) -> np.ndarray:
    """Range-time matrix of the returns for a staggered-PRI radar (no noise).

    Returns are injected by absolute arrival time into one continuous
    recording, exactly as for a constant PRI, and each pulse's listening
    window is then cut from it starting at that pulse's transmit sample.  The
    window (range gate) is the shortest PRI, so every pulse has the same
    number of range bins.  Echoes from beyond the gate are not modelled
    separately: they land in a later window like any other sample.

    Args:
        waveform: WaveformSample with the sampled pulse set.
        return_list: A list of Return and Clutter objects.
        radar: Radar with ``pri_stagger`` set.

    Returns:
        np.ndarray: Complex RTM, shape ``(min PRI in samples, n_pulses)``.
    """
    tx_samples = np.round(radar.pulse_tx_times * radar.sample_rate).astype(int)
    pris = radar.stagger_pri_samples
    gate = pris.min()
    # An (n, 1) array flattens to the recording itself, so the flat-buffer
    # injection used for constant-PRI datacubes works unchanged.
    recording = np.zeros(
        (tx_samples[-1] + pris[(radar.n_pulses - 1) % len(pris)], 1), np.complex64
    )
    add_returns(recording, waveform, return_list, radar)
    return recording[tx_samples[None, :] + np.arange(gate)[:, None], 0]
