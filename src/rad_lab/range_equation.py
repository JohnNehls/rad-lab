"""Radar and one-way link range equations.

Implements the two-way radar range equation (received signal power and SNR)
and the one-way Friis transmission equation used for jammer link budgets.
"""

from . import constants as c


def signal_range_eqn(
    Pt: float, Gt: float, Gr: float, sigma: float, wavelength: float, R: float, L: float
) -> float:
    """
    Calculate the received signal power for a radar system.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        L (float): System losses [unitless]

    Returns:
        float: Received signal power [W]
    """
    return (Pt * Gt * Gr * sigma * wavelength**2) / (((4 * c.PI) ** 3) * (R**4) * L)


def signal_range_eqn_one_way(
    Pt: float, Gt: float, Gr: float, wavelength: float, R: float, L: float
) -> float:
    """
    Calculate the received power for a one-way communication link (Friis equation).

    Models a transmitter and receiver with no target reflection — e.g. an EA
    platform retransmitting a stored pulse directly to the radar receiver.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range between transmitter and receiver [m]
        L (float): System losses [unitless]

    Returns:
        float: Received power [W]
    """
    return (Pt * Gt * Gr * wavelength**2) / ((4 * c.PI) ** 2 * R**2 * L)


def noise_power(B: float, F: float, T: float) -> float:
    """
    Calculate the thermal noise power of the receiver.

    Args:
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        T (float): System noise temperature [Kelvin]

    Returns:
        float: Noise power [W]
    """
    return c.K_BOLTZ * T * B * F


def snr_range_eqn_uncoded(
    Pt: float,
    Gt: float,
    Gr: float,
    sigma: float,
    wavelength: float,
    R: float,
    B: float,
    F: float,
    L: float,
    T: float,
) -> float:
    """
    Calculate the single-pulse Signal-to-Noise Ratio (SNR) for an uncoded pulse.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]

    Returns:
        float: Signal-to-Noise Ratio [unitless]
    """
    return (Pt * Gt * Gr * sigma * wavelength**2) / (
        ((4 * c.PI) ** 3) * (R**4) * c.K_BOLTZ * T * B * F * L
    )


def snr_range_eqn(
    Pt: float,
    Gt: float,
    Gr: float,
    sigma: float,
    wavelength: float,
    R: float,
    B: float,
    F: float,
    L: float,
    T: float,
    time_bandwidth_prod: float,
) -> float:
    """
    Calculate the single-pulse Signal-to-Noise Ratio (SNR) for a pulse with pulse compression.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        time_bandwidth_prod (float): Pulse compression ratio (time-bandwidth product) [unitless]

    Returns:
        float: Signal-to-Noise Ratio [unitless]
    """
    return (
        snr_range_eqn_uncoded(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T) * time_bandwidth_prod
    )


def snr_range_eqn_cp(
    Pt: float,
    Gt: float,
    Gr: float,
    sigma: float,
    wavelength: float,
    R: float,
    B: float,
    F: float,
    L: float,
    T: float,
    n_p: float,
    time_bandwidth_prod: float,
) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) after coherent processing of multiple pulses.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        n_p (float): Number of pulses coherently integrated [unitless]
        time_bandwidth_prod (float): Pulse compression ratio [unitless]

    Returns:
        float: Integrated Signal-to-Noise Ratio [unitless]
    """
    single_pulse_snr = snr_range_eqn(
        Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, time_bandwidth_prod
    )
    return single_pulse_snr * n_p


def snr_range_eqn_bpsk_cp(
    Pt: float,
    Gt: float,
    Gr: float,
    sigma: float,
    wavelength: float,
    R: float,
    B: float,
    F: float,
    L: float,
    T: float,
    n_p: float,
    n_c: float,
) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) for Binary Phase Shift Keying (BPSK) pulses after coherent processing.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        n_p (float): Number of pulses coherently integrated [unitless]
        n_c (float): Number of binary chips per pulse [unitless]

    Returns:
        float: Integrated Signal-to-Noise Ratio [unitless]
    """
    return snr_range_eqn_cp(Pt, Gt, Gr, sigma, wavelength, R, B, F, L, T, n_p, n_c)


def snr_range_eqn_duty_factor_pulses(
    Pt: float,
    Gt: float,
    Gr: float,
    sigma: float,
    wavelength: float,
    R: float,
    F: float,
    L: float,
    T: float,
    Tcpi: float,
    tau_df: float,
) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) using the duty factor and coherent processing interval.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Peak transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        R (float): Range to the target [m]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        Tcpi (float): Total coherent processing interval (CPI) duration [s]
        tau_df (float): Radar duty factor [0 to 1]

    Returns:
        float: Integrated Signal-to-Noise Ratio [unitless]
    """
    assert 0 <= tau_df <= 1, "duty factor must be in [0,1]."

    single_pulse_snr = snr_range_eqn_uncoded(Pt, Gt, Gr, sigma, wavelength, R, 1, F, L, T)
    return single_pulse_snr * Tcpi * tau_df


def max_target_detection_range(
    Pt: float,
    Gt: float,
    Gr: float,
    sigma: float,
    wavelength: float,
    SNR_thresh: float,
    B: float,
    F: float,
    L: float,
    T: float,
) -> float:
    """
    Calculate the maximum detectable range for a single uncoded pulse given an SNR threshold.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        SNR_thresh (float): Minimum SNR required for detection [unitless]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]

    Returns:
        float: Maximum detection range [m]
    """
    return (
        (Pt * Gt * Gr * sigma * wavelength**2)
        / (((4 * c.PI) ** 3) * (SNR_thresh) * c.K_BOLTZ * T * B * F * L)
    ) ** (1 / 4)


def max_target_detection_range_bpsk_cp(
    Pt: float,
    Gt: float,
    Gr: float,
    sigma: float,
    wavelength: float,
    SNR_thresh: float,
    B: float,
    F: float,
    L: float,
    T: float,
    n_p: float,
    n_c: float,
) -> float:
    """
    Calculate the maximum detectable range for coherently processed BPSK pulses.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        SNR_thresh (float): Minimum SNR required for detection [unitless]
        B (float): Receiver bandwidth [Hz]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        n_p (float): Number of pulses coherently processed [unitless]
        n_c (float): Number of binary chips per pulse [unitless]

    Returns:
        float: Maximum detection range [m]
    """
    one_pulse = max_target_detection_range(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, B, F, L, T)
    return one_pulse * (n_p * n_c) ** (1 / 4)


def max_target_detection_range_dutyfactor_cp(
    Pt: float,
    Gt: float,
    Gr: float,
    sigma: float,
    wavelength: float,
    SNR_thresh: float,
    F: float,
    L: float,
    T: float,
    Tcpi: float,
    tau_df: float,
) -> float:
    """
    Calculate the maximum detectable range for coherently processed pulses using duty factor parameters.
    All gain and loss factors must be in linear (unitless) form.

    Args:
        Pt (float): Peak transmit power [W]
        Gt (float): Transmit antenna gain [unitless]
        Gr (float): Receive antenna gain [unitless]
        sigma (float): Radar cross section [m^2]
        wavelength (float): Wavelength of the carrier signal [m]
        SNR_thresh (float): Minimum SNR required for detection [unitless]
        F (float): Receiver noise factor [unitless]
        L (float): System losses [unitless]
        T (float): System noise temperature [Kelvin]
        Tcpi (float): Total coherent processing interval duration [s]
        tau_df (float): Radar duty factor [0 to 1]

    Returns:
        float: Maximum detection range [m]
    """
    one_pulse = max_target_detection_range(Pt, Gt, Gr, sigma, wavelength, SNR_thresh, 1, F, L, T)
    return one_pulse * (Tcpi * tau_df) ** (1 / 4)
