"""Signal processing and unit conversion utilities.

Miscellaneous helpers: dB/linear conversions, phase wrapping to [−π, π),
replacing zeros with the smallest float, and other small signal utilities
used across the library.
"""

import numpy as np
from . import constants as c


def power2db(power: float) -> float:
    """Converts power from watts to decibels (dB).

    The conversion is calculated as 10 * log10(power).

    Args:
        power (float): Power in watts (W).

    Returns:
        float: Power in decibels (dB).
    """
    return 10 * np.log10(power)


def db2power(db: float) -> float:
    """Converts power from decibels (dB) to watts.

    The conversion is calculated as 10**(db / 10).

    Args:
        db (float): Power in decibels (dB).

    Returns:
        float: Power in watts (W).
    """
    return 10 ** (db / 10)


def volt2db(voltage: float) -> float:
    """Converts a voltage or amplitude ratio to decibels (dB).

    The conversion is calculated as 20 * log10(voltage). This is typically
    used for field quantities like voltage or pressure.

    Args:
        voltage (float): Voltage or amplitude ratio (unitless).

    Returns:
        float: The corresponding value in decibels (dB).
    """
    return 20 * np.log10(voltage)


def db2volt(db: float) -> float:
    """Converts a decibel (dB) value to a voltage or amplitude ratio.

    The conversion is calculated as 10**(db / 20).

    Args:
        db (float): A value in decibels (dB).

    Returns:
        float: The corresponding voltage or amplitude ratio (unitless).
    """
    return 10 ** (db / 20)


def phase_negpi_pospi(phase: float | np.ndarray) -> np.ndarray:
    """Wraps phase angles to the interval [-pi, pi).

    Args:
        phase (float or array-like): Phase angle(s) in radians.

    Returns:
        numpy.ndarray: Phase angle(s) wrapped to the [-pi, pi) interval.
    """

    phase = np.atleast_1d(np.array(phase))
    phase = phase % (2 * c.PI)
    phase[phase >= c.PI] -= 2 * c.PI
    return phase


def phase_zero_twopi(phase: float | np.ndarray) -> np.ndarray:
    """Wraps phase angles to the interval [0, 2*pi).

    Args:
        phase (float or array-like): Phase angle(s) in radians.

    Returns:
        numpy.ndarray: Phase angle(s) wrapped to the [0, 2*pi) interval.
    """
    phase = np.array(phase)
    phase = phase % (2 * c.PI)
    return phase


def zero_to_smallest_float(array: np.ndarray, value: float = 1e-16) -> None:
    """Replaces all zero elements in a NumPy array with a small float value.

    This operation is performed in-place. It is often used to avoid
    division-by-zero errors or issues with taking logarithms of zero.

    Args:
        array (numpy.ndarray): The input array to modify.
        value (float, optional): The small value to substitute for zeros.
            Defaults to 1e-16.

    Returns:
        None: The array is modified in-place.
    """
    array[array == 0] = value
