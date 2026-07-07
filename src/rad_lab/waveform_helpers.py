"""Waveform injection and spectral plotting utilities.

Low-level helpers for inserting a pulse waveform into a flat datacube buffer,
applying a matched filter via FFT convolution, resampling waveforms, and
plotting pulse shapes with their spectra.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy import fft
from scipy.interpolate import interp1d
from scipy import signal

logger = logging.getLogger(__name__)


def zeropad_waveform(
    t: np.ndarray, waveform: np.ndarray, N_pad: int
) -> tuple[np.ndarray, np.ndarray]:
    """Zeropads a waveform and adjusts the corresponding time array.

    This function appends a specified number of zeros to the end of a waveform
    and recalculates the time array to match the new length, preserving the
    original time step.

    Args:
        t (np.ndarray): The original 1D time array.
        waveform (np.ndarray): The 1D signal array.
        N_pad (int): The number of zero samples to append.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: The new, extended time array.
            - np.ndarray: The new, zero-padded waveform.
    """
    assert isinstance(N_pad, int)
    assert len(t) == len(waveform)
    waveform = np.append(waveform, np.zeros(N_pad))
    t = np.arange(waveform.size) * (t[1] - t[0]) + t[0]
    return t, waveform


def moving_average(waveform: np.ndarray, N_elements: int) -> np.ndarray:
    """Calculates the moving average of a waveform.

    This function smooths the input waveform by convolving it with a uniform
    kernel of a specified size.

    Args:
        waveform (np.ndarray): The 1D signal array to be averaged.
        N_elements (int): The number of elements in the moving average window.

    Returns:
        np.ndarray: The smoothed waveform as a 1D array.
    """
    assert isinstance(N_elements, int)
    kernel = np.ones(N_elements) / N_elements
    ave = np.convolve(waveform, kernel, mode="same")
    return ave


def find_width(
    x: np.ndarray,
    y: np.ndarray,
    interp_max: int = 5,
    interp_count: int = 0,
    interp_scale: int = 2,
) -> tuple[float, float, float] | list[float]:
    """Recursively finds the full width at half maximum (FWHM) of a signal.

    This function calculates the width of a pulse by finding the points where the
    signal crosses half of its maximum amplitude. If two such points are not
    found at the current resolution, it recursively interpolates the signal to a
    finer grid and tries again, up to a maximum number of interpolations.

    Args:
        x (np.ndarray): The independent variable array (e.g., time or frequency).
        y (np.ndarray): The dependent variable array (the signal).
        interp_max (int, optional): The maximum number of recursive interpolation
            steps. Defaults to 5.
        interp_count (int, optional): The current interpolation step count, used
            for recursion. Defaults to 0.
        interp_scale (int, optional): The factor by which to increase the number
            of points during each interpolation step. Defaults to 2.

    Returns:
        tuple[float, float, float] or list[float]: A tuple containing:
            - float: The calculated pulse width (FWHM).
            - float: The x-value at the start of the pulse width (first
              half-max crossing).
            - float: The x-value at the end of the pulse width (last
              half-max crossing).
            Returns [np.nan, np.nan, np.nan] if the width cannot be found within
            the interpolation limit.
    """
    # Step 1: Find the maximum amplitude of the signal
    max_amplitude = np.max(y)

    # Step 2: Calculate the half maximum amplitude
    half_max_amplitude = max_amplitude / 2

    # Step 3: Find the points in the signal where the y is equal to the half maximum amplitude
    ind = np.where(np.isclose(y, half_max_amplitude, rtol=1e-2))[0]

    # base case
    if ind.size >= 2:
        t_start = x[ind[0]]
        t_end = x[ind[-1]]
        pulse_width = t_end - t_start
        return pulse_width, t_start, t_end

    # stop infinite recursion
    elif interp_count >= interp_max:
        logger.error("Error: cannot find width")
        return [np.nan, np.nan, np.nan]

    # recursive step
    else:
        logger.info(
            f"find_width: interpolating to {interp_scale}x the sample rate {interp_count=}"
        )
        interp_func = interp1d(x, y, kind="linear")
        newx = np.linspace(x[0], x[-1], x.size * interp_scale)
        newy = interp_func(newx)
        return find_width(newx, newy, interp_max, interp_count + 1, interp_scale)


def plot_pulse_and_spectrum(
    t: np.ndarray,
    mag: np.ndarray,
    title: str | None = None,
    n_pad: int = 0,
    print_bandwidth: bool = True,
    spec_dec: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plots a signal in the time and frequency domains.

    Generates a two-panel plot showing the signal's magnitude over time and its
    frequency spectrum. The spectrum is calculated using an FFT, and the
    signal can be zero-padded to improve frequency resolution.

    Args:
        t (np.ndarray): The time array for the signal.
        mag (np.ndarray): The signal magnitude array (can be complex).
        title (str, optional): The super-title for the entire figure.
            Defaults to None.
        n_pad (int, optional): The number of zeros to append for the FFT
            calculation, improving frequency resolution. Defaults to 0.
        print_bandwidth (bool, optional): If True, calculates and prints the
            signal's bandwidth (FWHM of the spectrum). Defaults to True.
        spec_dec (bool, optional): If True, plots the spectrum magnitude in
            decibels (dB). Defaults to False.

    Returns:
        tuple[plt.Figure, np.ndarray[plt.Axes]]: A tuple containing:
            - plt.Figure: The matplotlib Figure object.
            - np.ndarray[plt.Axes]: An array of the two matplotlib Axes objects
              (time domain and frequency domain).
    """
    ## time domain ##
    fig, ax = plt.subplots(1, 2)
    if np.iscomplexobj(mag):
        ax[0].plot(t, np.abs(mag), "-o", label="magnitude")
        ax[0].plot(t, np.real(mag), "--o", label="real")
        ax[0].plot(t, np.imag(mag), "-.o", label="imag")
        ax[0].legend()
    else:
        ax[0].plot(t, mag, "-o")
        ax[0].set_ylim((min(0, mag.min() * 1.1), mag.max() * 1.1))

    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("baseband signal")
    ax[0].grid()

    ##  frequency domain ##
    # pad with zeros for greater freq resolution
    mag = np.append(mag, np.zeros(n_pad))
    dt = t[1] - t[0]
    N = mag.size

    MAG = fft.fftshift(fft.fft(mag)) / N
    f = fft.fftshift(fft.fftfreq(N, dt))

    val = abs(MAG)
    val = val / val.max()
    if spec_dec:
        val = 20 * np.log10(val)
        ax[1].set_ylabel("baseband magnitude dB")
    else:
        ax[1].set_ylabel("baseband magnitude")
    ax[1].plot(f, val, "-")
    ax[1].set_xlabel("frequency [Hz]")

    ax[1].grid()

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if print_bandwidth:
        print("\tbandwidth:")
        PW, f_start, f_end = find_width(f, abs(MAG))
        print(f"\t{PW=:.1e} {f_start=:.1e} {f_end=:.1e}")

    return fig, ax


def autocorrelate_waveform(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the autocorrelation of a waveform using the FFT method.

    The autocorrelation is computed by multiplying the Fourier Transform of the
    waveform with its complex conjugate and then performing an inverse Fourier
    Transform.

    Args:
        waveform (np.ndarray): The 1D input signal (real or complex).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: The autocorrelation result.
            - np.ndarray: An array of index shifts corresponding to the lags
              of the autocorrelation, centered at zero.
    """
    Nwf = waveform.size
    Nfft = 2 * Nwf - 1  # add some padding
    WF = fft.fft(waveform, Nfft)
    autoCor = fft.ifft(WF * np.conj(WF))
    autoCor = fft.fftshift(autoCor)
    index_shift = np.arange(-(Nwf - 1), Nwf)

    return autoCor, index_shift


def plot_pulse_and_xcorrelation(
    t: np.ndarray,
    mag: np.ndarray,
    title: str | None = None,
    print_width: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plots a signal and its autocorrelation.

    Generates a two-panel plot showing the signal's magnitude over time and its
    autocorrelation function.

    Args:
        t (np.ndarray): The time array for the signal.
        mag (np.ndarray): The signal magnitude array (can be complex).
        title (str, optional): The super-title for the entire figure.
            Defaults to None.
        print_width (bool, optional): If True, calculates and prints the width
            (FWHM) of the main autocorrelation lobe. Defaults to True.

    Returns:
        tuple[plt.Figure, np.ndarray[plt.Axes]]: A tuple containing:
            - plt.Figure: The matplotlib Figure object.
            - np.ndarray[plt.Axes]: An array of the two matplotlib Axes objects
              (time domain and autocorrelation).
    """
    dt = t[1] - t[0]

    fig, ax = plt.subplots(1, 2)
    if np.iscomplexobj(mag):
        ax[0].plot(t, np.abs(mag), "-o", label="magnitude")
        ax[0].plot(t, np.real(mag), "--o", label="real")
        ax[0].plot(t, np.imag(mag), "-.o", label="imag")
        ax[0].legend()

    else:
        ax[0].plot(t, mag, "-o")

    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("baseband signal")
    ax[0].grid()

    xcor, index_shift = autocorrelate_waveform(mag)
    time_shift = index_shift * dt
    val = abs(xcor)
    val = val / val.max()
    ax[1].plot(time_shift, val, "-o")
    ax[1].set_xlabel("time shift [s]")
    ax[1].set_ylabel("cross correlation mag")
    ax[1].grid()

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if print_width:
        print("\txcor:")
        PW, f_start, f_end = find_width(time_shift, abs(xcor))
        print(f"\t{PW=:.1f} {f_start=:.1f} {f_end=:.1f}")
        PW_av, f_start_av, f_end_av = find_width(time_shift, moving_average(abs(xcor), 3))
        print(f"\t{PW_av=:.1f} {f_start_av=:.1f} {f_end_av=:.1f}")

    return fig, ax


def add_waveform_at_index(ar: np.ndarray, waveform: np.ndarray, index: int) -> np.ndarray:
    """Adds a waveform to an array at a specified starting index.

    This function modifies the target array `ar` in-place by adding the
    `waveform` to it, starting at the given `index`. If the waveform extends
    beyond the bounds of `ar`, it is truncated.

    Args:
        ar (np.ndarray): The target array to be modified.
        waveform (np.ndarray): The waveform to add to the target array.
        index (int): The starting index in `ar` where the waveform will be
            added.

    Returns:
        np.ndarray: The modified target array `ar`.
    """
    Nar = ar.size
    Nwv = waveform.size

    n_add = min(Nwv, max(0, Nar - index))
    ar[index : index + n_add] += waveform[:n_add]
    if n_add < Nwv:
        logger.info(f"add_waveform_at_index: waveform truncated \n\t{index}\n\t{Nar}\n\t{Nwv}")
    return ar


def matchfilter_with_waveform(
    ar: np.ndarray, waveform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Performs matched filtering of a signal with a given waveform.

    This function applies a matched filter to the input array `ar` using the
    provided `waveform` as the template. The filter is implemented as a
    convolution with the time-reversed complex conjugate of the waveform.

    Args:
        ar (np.ndarray): The input signal array to be filtered.
        waveform (np.ndarray): The template waveform for the matched filter.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: An array of index shifts corresponding to the output,
              centered at zero.
            - np.ndarray: The result of the matched filter convolution.
    """
    Nar = ar.size
    kernel = np.conj(waveform)[::-1]
    conv = signal.convolve(ar, kernel, mode="same", method="direct")
    index_shift = np.arange(Nar) - Nar // 2

    return index_shift, conv
