"""Amplitude monopulse angle estimation.

Implements the sum-and-difference amplitude monopulse technique for estimating
target angle from a two-element array, including peak-finding across a
range-Doppler map.
"""

import numpy as np


def amplitude_monopulse(sig_a: np.ndarray, sig_b: np.ndarray, dx: float) -> np.ndarray:
    """Estimates sin(theta) at each sample or bin via amplitude monopulse ratio.

    Computes the standard sum/difference monopulse estimator:
        v_theta = arctan(2 * Im(delta/sum)) / (2*pi*dx)

    where sum = sig_a + sig_b and delta = sig_a - sig_b.

    Note:
        This is a small-angle estimator.  Im(delta/sum) equals
        tan(pi*dx*sin(theta)) exactly, so the arctan(2*x) linearisation
        introduces a bias that grows off boresight (about 0.1 deg at
        theta = 5 deg for dx = 0.5).

    Args:
        sig_a: Complex signal from the first array element.
        sig_b: Complex signal from the second array element.
        dx: Element separation in wavelengths.

    Returns:
        v_theta (np.ndarray): Estimated sin(theta) at each sample or bin.
    """
    rho = 2 * np.pi * dx
    sum_ = sig_a + sig_b
    delta = sig_a - sig_b
    return np.arctan(2 * (delta / sum_).imag) / rho


def monopulse_angle_deg(sig_a: np.ndarray, sig_b: np.ndarray, dx: float) -> np.ndarray:
    """Estimates target angle in degrees at each sample or bin.

    Args:
        sig_a: Complex signal from the first array element.
        sig_b: Complex signal from the second array element.
        dx: Element separation in wavelengths.

    Returns:
        np.ndarray: Estimated angle in degrees at each sample or bin.
    """
    return np.rad2deg(np.arcsin(amplitude_monopulse(sig_a, sig_b, dx)))


def monopulse_angle_at_peak_deg(sig_a: np.ndarray, sig_b: np.ndarray, dx: float) -> float:
    """Estimates target angle in degrees at the peak power bin.

    Finds the index of maximum magnitude in sig_a, then returns the
    monopulse angle estimate at that bin. Suitable for use on RDMs or
    frequency-domain signals where monopulse is only meaningful at bins
    containing target power.

    Args:
        sig_a: Complex signal from the first array element (used for peak detection).
        sig_b: Complex signal from the second array element.
        dx: Element separation in wavelengths.

    Returns:
        float: Estimated angle in degrees at the peak bin.
    """
    theta = monopulse_angle_deg(sig_a, sig_b, dx)
    peak_index = np.unravel_index(np.argmax(np.abs(sig_a)), sig_a.shape)
    return float(theta[peak_index])
