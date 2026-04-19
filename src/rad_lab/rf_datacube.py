"""RF datacube creation and processing.

Provides helpers to allocate a pulse-Doppler datacube, compute range and
frequency axes, apply a matched filter via fast convolution, and Doppler-process
the slow-time dimension with an FFT.
"""

import numpy as np
from scipy import fft, signal
from . import constants as c
from .waveform_helpers import matchfilter_with_waveform


def range_axis(fs: float, N_r: int) -> np.ndarray:
    """Generates the range axis for a radar datacube.

    This function calculates the range corresponding to each range bin
    based on the sampling frequency. The range resolution is determined
    by the speed of light and the sampling rate.

    Args:
        fs (float): The sampling frequency in Hertz [Hz].
        N_r (int): The number of range bins (samples in fast-time).

    Returns:
        np.ndarray: A 1D NumPy array representing the range axis in meters [m].
    """
    dR_grid = c.C / (2 * fs)
    R_axis = np.arange(1, N_r + 1) * dR_grid  # Process fast time
    return R_axis


def number_range_bins(fs: float, prf: float) -> int:
    """Calculates the number of range bins.

    The number of range bins is determined by the number of samples collected
    during one pulse repetition interval (PRI). PRI is the reciprocal of the
    pulse repetition frequency (PRF).

    Args:
        fs (float): The sampling frequency [Hz].
        prf (float): The pulse repetition frequency [Hz].

    Returns:
        int: The total number of range bins.
    """
    return int(fs / prf)


def data_cube(fs: float, prf: float, N_p: int) -> np.ndarray:
    """Creates an empty, complex-valued datacube.

    This function initializes a 2D NumPy array (datacube) with zeros,
    representing the raw data collected over a coherent processing interval (CPI).
    The dimensions are determined by the number of range bins and the number of pulses.

    Args:
        fs (float): The sampling frequency in Hertz [Hz].
        prf (float): The pulse repetition frequency in Hertz [Hz].
        N_p (int): The number of pulses in the coherent processing interval (CPI).

    Returns:
        np.ndarray: A 2D NumPy array of shape (N_range_bins, N_pulses)
                    initialized with complex zeros.
    """
    Nr = number_range_bins(fs, prf)
    dc = np.zeros((Nr, N_p), dtype=np.complex64)
    return dc


def doppler_process(datacube: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Performs Doppler processing on a radar datacube.

    This function applies a Fast Fourier Transform (FFT) across the slow-time
    (pulse) dimension of the datacube to transform the data into the
    Range-Doppler domain. The operation is performed in-place on the input
    datacube. It also generates the corresponding Doppler frequency and range axes.

    Args:
        datacube (np.ndarray): A 2D NumPy array representing the time-domain
                             datacube, with shape (N_range_bins, N_pulses).
                             This array will be modified in-place.
        fs (float): The sampling frequency in Hertz [Hz].

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - f_axis (np.ndarray): The Doppler frequency axis, [-PRF/2, PRF/2) [Hz].
            - R_axis (np.ndarray): The range axis [delta_r, R_ambigious] [m].
    """
    N_r, N_p = datacube.shape
    dR_grid = c.C / (2 * fs)
    prf = fs / datacube.shape[0]
    R_axis = np.arange(1, N_r + 1) * dR_grid  # Process fast time
    f_axis = fft.fftshift(fft.fftfreq(N_p, 1 / prf))  # process slow time
    datacube[:] = fft.fftshift(fft.fft(datacube, axis=1), axes=1)
    return f_axis, R_axis


def matchfilter(
    datacube: np.ndarray, pulse_wvf: np.ndarray, sample_rate: float, pedantic: bool = True
) -> None:
    """Applies a matched filter to a datacube for pulse compression.

    This is a discrete approximation of the continuous-time correlator
    ``y(t) = ∫ s(τ) p*(τ-t) dτ``: each output sample is the discrete sum
    scaled by ``Δt = 1 / sample_rate``.  With a unit-amplitude transmit
    pulse (``|p|=1`` over duration ``T``) this gives a peak output voltage
    of ``V_rx · T`` and an output noise variance of ``R · N₀ · T``, so the
    range-equation TB gain emerges from the SNR ratio without any explicit
    rescaling of the pulse template.

    Two implementations are available:
    - Pedantic (True): Iteratively applies the matched filter to each pulse
      using a time-domain helper function. This is typically slower but can
      be clearer to understand.
    - Non-pedantic (False): Uses a more efficient frequency-domain approach
      by performing convolution via FFT. This involves a single FFT of the
      waveform kernel and is generally faster for large datacubes.

    Args:
        datacube: 2D time-domain datacube with shape (N_range_bins, N_pulses),
            modified in-place.
        pulse_wvf: 1D transmitted pulse template (unit-amplitude convention,
            see :class:`rad_lab.waveform.WaveformSample`).
        sample_rate: ADC sample rate [Hz].  Used to scale the output by
            ``Δt = 1/sample_rate``.
        pedantic: If True, use the iterative time-domain helper; if False,
            use FFT-based convolution.  Defaults to True.

    Returns:
        None: The `datacube` is modified in-place.
    """
    dt = 1.0 / sample_rate
    if pedantic:
        for j in range(datacube.shape[1]):
            _, mf = matchfilter_with_waveform(datacube[:, j], pulse_wvf)
            datacube[:, j] = mf * dt
    else:
        kernel = np.conj(pulse_wvf)[::-1]
        datacube[:] = signal.fftconvolve(datacube, kernel.reshape(-1, 1), mode="same", axes=0)
        datacube *= dt
