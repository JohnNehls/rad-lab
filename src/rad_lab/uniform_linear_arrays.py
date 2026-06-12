"""Uniform linear array (ULA) gain patterns and steering vectors.

Provides functions to compute element steering vectors, apply inter-element
time delays, and plot one-way and two-way array gain patterns as a function
of angle.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import matplotlib.pyplot as plt
from . import constants as c
from scipy import interpolate


def steering_vector(el_pos: np.ndarray, theta: float) -> np.ndarray:
    """
    Computes the Vandermonde steering vector for a linear array.

    This vector represents the phase shifts of a planar wave arriving from a specific
    angle as measured at each element of the array.

    Args:
        el_pos (np.ndarray): A 1D array of antenna element positions, normalized
                             by the signal wavelength.
        theta (float): The steering angle in degrees, where 0 is broadside.

    Returns:
        np.ndarray: A complex-valued steering vector of the same size as `el_pos`.
    """
    theta_rad = np.deg2rad(theta)
    return np.exp(-1j * 2 * np.pi * np.sin(theta_rad) * el_pos)


def linear_antenna_gain(
    el_pos: np.ndarray,
    weight_vec: np.ndarray | None = None,
    N_theta: int = 10000,
    steer_angle: float = 0,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the complex voltage gain pattern for a linear antenna array.

    Args:
        el_pos (np.ndarray): 1D array of element positions, normalized by the
                             signal wavelength.
        weight_vec (np.ndarray, optional): A vector of complex weights for each
            antenna element. If None, uniform weights (all ones) are used.
            Defaults to None.
        N_theta (int, optional): The number of angular points to calculate the
            gain over. Defaults to 10000.
        steer_angle (float, optional): The angle in degrees at which to steer
            the main beam. 0 degrees is broadside. Defaults to 0.
        plot (bool, optional): If True, plots the gain pattern in dBi.
            Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - theta_vec (np.ndarray): The grid of angles in degrees, from -90 to 90.
            - gain_vec (np.ndarray): The complex voltage gain at each angle in `theta_vec`.
    """
    if weight_vec is None:
        weight_vec = np.ones(len(el_pos))

    theta_grid = np.linspace(-np.pi / 2, np.pi / 2, N_theta)

    steer_vec = steering_vector(el_pos, steer_angle)
    weight_vec = weight_vec * steer_vec

    A = np.exp(1j * 2 * np.pi * np.outer(np.sin(theta_grid), el_pos))
    Af = A @ weight_vec

    # The difference between the antenna's gain and the isotropic antenna's gain is dBi
    af_dbi = 20 * np.log10(abs(Af))

    if plot:
        plt.figure()
        plt.plot(np.rad2deg(theta_grid), af_dbi)
        plt.ylim((-30, af_dbi.max() * 1.2))
        plt.xlabel(r"Angle $\theta$ [deg]")
        plt.ylabel("Gain [dBi]")
        plt.grid()

    return np.rad2deg(theta_grid), Af


def linear_antenna_gain_meters(
    el_pos: np.ndarray,
    fc: float,
    weight_vec: np.ndarray | None = None,
    N_theta: int = 10000,
    steer_angle: float = 0,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates gain pattern from element positions in meters and frequency.

    This is a convenience wrapper for `linear_antenna_gain` that converts
    physical positions and frequency into wavelength-normalized positions.

    Args:
        el_pos (np.ndarray): 1D array of element positions in meters.
        fc (float): The signal's center frequency in Hertz.
        weight_vec (np.ndarray, optional): A vector of complex weights for each
            antenna element. If None, uniform weights are used. Defaults to None.
        N_theta (int, optional): The number of angular points to calculate the
            gain over. Defaults to 10000.
        steer_angle (float, optional): The angle in degrees at which to steer
            the main beam. 0 degrees is broadside. Defaults to 0.
        plot (bool, optional): If True, plots the gain pattern in dBi.
            Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - theta_vec (np.ndarray): The grid of angles in degrees, from -90 to 90.
            - gain_vec (np.ndarray): The complex voltage gain at each angle.
    """
    wavelength = c.C / fc
    el_pos_per_wl = el_pos / wavelength
    return linear_antenna_gain(
        el_pos_per_wl, weight_vec=weight_vec, N_theta=N_theta, steer_angle=steer_angle, plot=plot
    )


def linear_antenna_gain_N_db(
    N_el: int,
    dx: float,
    weight_vec: np.ndarray | None = None,
    N_theta: int = 10000,
    steer_angle: float = 0,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates gain pattern in dBi for a uniform linear array.

    This function defines the array geometry based on the number of elements
    and their uniform spacing. It then computes and returns the gain in dBi.

    Args:
        N_el (int): The number of antenna array elements.
        dx (float): The spacing between elements, normalized by signal wavelength.
        weight_vec (np.ndarray, optional): A vector of complex weights for each
            antenna element. If None, uniform weights are used. Defaults to None.
        N_theta (int, optional): The number of angular points to calculate the
            gain over. Defaults to 10000.
        steer_angle (float, optional): The angle in degrees at which to steer
            the main beam. 0 degrees is broadside. Defaults to 0.
        plot (bool, optional): If True, plots the gain pattern in dBi.
            Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - theta_vec (np.ndarray): The grid of angles in degrees, from -90 to 90.
            - gain_vec_db (np.ndarray): The voltage gain in dBi at each angle.
    """
    L = (N_el - 1) * dx
    el_pos = np.linspace(-L / 2, L / 2, N_el)  # wavelengths

    thetas, gain = linear_antenna_gain(
        el_pos, weight_vec=weight_vec, N_theta=N_theta, steer_angle=steer_angle, plot=plot
    )
    return thetas, 20 * np.log10(abs(gain))


def array_phase_center(position_ar: np.ndarray, weight_ar: np.ndarray) -> float:
    """
    Calculates the phase center of an antenna array.

    The phase center is the apparent point from which the radiation emanates.
    It is calculated as the weighted average of the element positions.

    Args:
        position_ar (np.ndarray): 1D array of element positions. The unit of
            the output will match the unit of this input (e.g., meters).
        weight_ar (np.ndarray): A vector of complex weights for each element.
            The magnitude of the weights is used in the calculation.

    Returns:
        float: The position of the array's phase center.
    """
    assert len(position_ar) == len(weight_ar)
    # unsure if the weight should be abs value or not
    return np.sum(abs(weight_ar) * position_ar) / np.sum(weight_ar)


def apply_timeshift_due_to_element_position(
    signal_ar: np.ndarray, fs: float, element_position: float, tgt_angle: float
) -> np.ndarray:
    """
    Applies a time shift to a signal based on element position and angle.

    This function models the delay or advance a signal experiences when arriving
    at an off-center antenna element from a specific target angle. The time
    shift is applied by resampling the signal using cubic interpolation.

    Args:
        signal_ar (np.ndarray): A complex time-series signal from a single
            antenna element.
        fs (float): The sampling frequency of the signal in Hertz.
        element_position (float): The element's position relative to the array's
            phase center in meters.
        tgt_angle (float): The angle of the target in degrees, where 0 is
            broadside.

    Returns:
        np.ndarray: The time-shifted complex signal.
    """
    range_diff = element_position * np.sin(np.deg2rad(tgt_angle))
    time_shift = range_diff / c.C
    time_ar = np.arange(len(signal_ar)) / fs
    shifted_time = time_ar + time_shift
    interp_fun = interpolate.interp1d(time_ar, signal_ar, kind="cubic", fill_value="extrapolate")
    shifted_signal = interp_fun(shifted_time)

    return shifted_signal


def ula_pattern(
    el_pos: np.ndarray,
    weight_vec: np.ndarray | None = None,
    two_way: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Creates a beam-pattern callable from a ULA for use with SAR spotlight.

    Precomputes the array factor via :func:`linear_antenna_gain`, normalises
    to unit peak, and returns a function that maps off-boresight angles
    (in radians) to amplitude weights.  The returned callable is directly
    compatible with the ``beam_pattern`` parameter of :func:`rad_lab.sar.gen`.

    Args:
        el_pos: Element positions normalised by wavelength.
        weight_vec: Optional complex element weights (default: uniform).
        two_way: If True (default), square the one-way voltage pattern to
            model the round-trip amplitude.

    Returns:
        A callable ``pattern(theta_rad) -> weights`` where *theta_rad* is
        an array of off-boresight angles [rad] and *weights* is a
        same-shape array of amplitude weights in [0, 1].
    """
    theta_deg, complex_gain = linear_antenna_gain(el_pos, weight_vec)
    norm_gain = np.abs(complex_gain) / np.abs(complex_gain).max()
    interp_fn = interpolate.interp1d(
        theta_deg, norm_gain, kind="linear", bounds_error=False, fill_value=0.0
    )

    def pattern(theta_rad: np.ndarray) -> np.ndarray:
        weights = interp_fn(np.rad2deg(theta_rad))
        if two_way:
            weights = weights**2
        return weights

    return pattern
