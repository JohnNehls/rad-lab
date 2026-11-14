"""Moving Target Indication (MTI) pulse cancellers.

An MTI canceller is a short FIR high-pass filter along slow time (the pulse
axis).  Its nulls at zero Doppler and at every multiple of the PRF suppress
stationary clutter, at the cost of blind speeds where targets are also
cancelled.  The order-``n`` canceller has binomial taps with alternating signs
(order 1: ``[1, -1]``, order 2: ``[1, -2, 1]``) and magnitude response
``|H(f)|^2 = (2 sin(pi f / prf))^(2n)``.

A staggered PRI (``pri_stagger``, integer ratios as on `Radar`) moves the
first true blind speed out by ``mean(pri_stagger) / gcd(pri_stagger)``.  The
taps are unchanged, but each output sees a different set of pulse spacings,
so the response is the average of ``|H_n(f)|^2`` over the stagger cycle.

### References

Richards, M. A., *Fundamentals of Radar Signal Processing*, 2nd ed.,
McGraw-Hill, 2014, Ch. 5 (Doppler Processing).
"""

import numpy as np
from scipy.special import comb
from . import constants as c


def canceller_coefficients(order: int) -> np.ndarray:
    """Taps of the order-``order`` binomial pulse canceller.

    Order 0 is the identity filter ``[1]`` (no MTI).

    Args:
        order: Canceller order (number of delays).

    Returns:
        np.ndarray: Taps ``h[k] = (-1)^k * C(order, k)``, ``k = 0..order``.
    """
    k = np.arange(order + 1)
    return (-1.0) ** k * comb(order, k)


def apply_canceller(datacube: np.ndarray, order: int) -> np.ndarray:
    """Filters each range bin along slow time with the pulse canceller.

    Computes ``y[n] = sum_k h[k] * x[n - k]`` along the pulse axis.  The first
    ``order`` pulses have no complete filter history and are dropped, so the
    output has ``n_pulses - order`` pulses.

    Args:
        datacube: 2D complex array, shape ``(n_range_bins, n_pulses)``.
        order: Canceller order.

    Returns:
        np.ndarray: Filtered datacube, shape ``(n_range_bins, n_pulses - order)``.
    """
    coeffs = canceller_coefficients(order)
    n_out = datacube.shape[1] - order
    filtered = np.zeros((datacube.shape[0], n_out), dtype=datacube.dtype)
    for k, h in enumerate(coeffs):
        filtered += h * datacube[:, order - k : order - k + n_out]
    return filtered


def stagger_pris(prf: float, pri_stagger: tuple[int, ...]) -> np.ndarray:
    """One stagger cycle of PRIs [s]: ``pri_stagger * T0`` with mean PRF ``prf``."""
    ratios = np.array(pri_stagger)
    return ratios / (prf * ratios.mean())


def frequency_response(
    order: int, prf: float, freqs: np.ndarray, pri_stagger: tuple[int, ...] | None = None
) -> np.ndarray:
    """Power response ``|H(f)|^2`` of the pulse canceller.

    With a staggered PRI, output ``n`` combines pulses at times
    ``t_n, t_(n-1), ...`` and has response
    ``H_n(f) = sum_k h[k] exp(-j 2 pi f (t_n - t_(n-k)))``.  The returned
    power response is ``|H_n|^2`` averaged over the stagger cycle, which is
    the mean power gain of a long pulse train.

    Args:
        order: Canceller order.
        prf: Pulse repetition frequency [Hz]; the mean PRF with ``pri_stagger``.
        freqs: Doppler frequencies at which to evaluate the response [Hz].
        pri_stagger: Optional PRI stagger ratios (see `Radar`).

    Returns:
        np.ndarray: Power gain at each frequency [linear].
    """
    coeffs = canceller_coefficients(order)
    pris = stagger_pris(prf, pri_stagger or (1,))  # a constant PRI is a one-ratio stagger

    # Pulse times covering every position in the cycle plus the filter history.
    times = np.concatenate([[0], np.cumsum(np.resize(pris, len(pris) + order - 1))])
    power = np.zeros(len(freqs))
    for n in range(order, order + len(pris)):
        response = np.zeros(len(freqs), dtype=complex)
        for k, h in enumerate(coeffs):
            response += h * np.exp(-2j * np.pi * freqs * (times[n] - times[n - k]))
        power += np.abs(response) ** 2
    return power / len(pris)


def blind_speeds(
    prf: float, fcar: float, n: int = 3, pri_stagger: tuple[int, ...] | None = None
) -> np.ndarray:
    """First ``n`` non-zero blind range rates of a canceller.

    A target is cancelled whenever its Doppler is a multiple of the PRF,
    i.e. at range rates ``v_k = k * c * prf / (2 * fcar)``.  With a staggered
    PRI it must be a multiple of every PRF at once: ``1 / (g * T0)`` with
    ``g = gcd(pri_stagger)``, which is ``mean(pri_stagger) / g`` times the
    mean PRF.

    Args:
        prf: Pulse repetition frequency [Hz]; the mean PRF with ``pri_stagger``.
        fcar: Carrier frequency [Hz].
        n: Number of blind speeds to return.
        pri_stagger: Optional PRI stagger ratios (see `Radar`).

    Returns:
        np.ndarray: Blind range rates ``v_1 .. v_n`` [m/s].
    """
    ratios = pri_stagger or (1,)
    blind_prf = prf * np.mean(ratios) / np.gcd.reduce(ratios)
    return np.arange(1, n + 1) * c.C * blind_prf / (2 * fcar)


def noise_gain(order: int) -> float:
    """White-noise power gain of the canceller: the sum of squared taps.

    Equals 2 for order 1 and 6 for order 2.  It is also the average gain for
    a target whose Doppler is uniformly distributed over the PRF.
    """
    return float(np.sum(canceller_coefficients(order) ** 2))


def improvement_factor(
    order: int,
    prf: float,
    fcar: float,
    range_rate_std: float,
    pri_stagger: tuple[int, ...] | None = None,
) -> float:
    """Theoretical MTI improvement factor for Gaussian-spectrum clutter.

    The improvement factor is the signal-to-clutter ratio out over in,
    averaged over target Doppler: ``I = noise_gain / clutter_gain``.  The
    clutter power gain is the canceller response averaged over the clutter's
    zero-mean Gaussian spectrum of standard deviation ``sigma_f``:
    ``clutter_gain = integral |H(f)|^2 S_c(f) df``.  Integrating ``|H|^2``
    (always >= 0) avoids the cancellation that a tap-correlation sum suffers
    for narrow spectra, and ``|H|^2`` being PRF-periodic accounts for any
    spectral aliasing.  For order 1 this equals ``1 / (1 - rho)`` with
    ``rho = exp(-2 * pi^2 * sigma_f^2 / prf^2)``.  With a staggered PRI the
    cycle-averaged response is used, so the result is the mean over outputs.

    Args:
        order: Canceller order.
        prf: Pulse repetition frequency [Hz]; the mean PRF with ``pri_stagger``.
        fcar: Carrier frequency [Hz].
        range_rate_std: Clutter range-rate standard deviation [m/s].
        pri_stagger: Optional PRI stagger ratios (see `Radar`).

    Returns:
        float: Improvement factor [linear]; ``inf`` for perfectly stationary
        clutter (``range_rate_std = 0``).
    """
    if range_rate_std == 0:
        return np.inf

    doppler_std = 2 * range_rate_std * fcar / c.C
    freqs = np.linspace(-8, 8, 4001) * doppler_std  # +/- 8 sigma covers the spectrum
    clutter_psd = np.exp(-(freqs**2) / (2 * doppler_std**2))
    clutter_psd /= clutter_psd.sum()  # unit clutter power in
    clutter_gain = np.sum(clutter_psd * frequency_response(order, prf, freqs, pri_stagger))
    return noise_gain(order) / clutter_gain
