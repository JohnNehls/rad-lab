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
            - R_axis (np.ndarray): The range axis `[delta_r, R_ambigious]` [m].
    """
    N_r, N_p = datacube.shape
    prf = fs / N_r
    R_axis = range_axis(fs, N_r)
    f_axis = fft.fftshift(fft.fftfreq(N_p, 1 / prf))  # process slow time
    datacube[:] = fft.fftshift(fft.fft(datacube, axis=1), axes=1)
    return f_axis, R_axis


def range_window(
    n_taps: int, window: str = "none", window_kwargs: dict | None = None
) -> np.ndarray:
    """Unit-mean amplitude taper for weighting a matched-filter replica.

    Weighting the replica across its ``N_taps`` samples suppresses the range
    sidelobes of the compressed pulse:

    - An LFM's frequency is linear in time, so a time taper tapers the
      swept-spectrum edges -- the frequency weighting Richards uses for
      range-sidelobe control.  The unweighted LFM sits at ~-13.2 dB peak
      sidelobes; weighting drives these down (e.g. Hamming ~-43 dB).
    - The cost is a broader mainlobe (coarser range resolution) and a small
      SNR loss from the now-mismatched filter.
    - Normalising to unit mean keeps the coherent peak gain at ``N_taps``, so
      the loss shows up as a raised noise floor, not a rescaled peak.

    Window vocabulary matches :func:`rad_lab._rdm_internals.create_window`.

    Args:
        n_taps: Length of the taper (number of replica samples).
        window: Window type.  One of ``"none"`` (rectangular, default),
            ``"chebyshev"`` (accepts ``window_kwargs={"at": <dB>}``,
            default 60), ``"blackman-harris"``, or ``"taylor"`` (accepts
            ``window_kwargs={"nbar": ..., "sll": ...}``).
        window_kwargs: Optional dict forwarded to the underlying
            ``scipy.signal.windows`` function.

    Returns:
        1D taper of length ``n_taps`` with mean 1.0.

    References:
        Richards, M. A., *Fundamentals of Radar Signal Processing*, 2nd ed.,
        McGraw-Hill, 2014, Ch. 4 (Radar Waveforms) — matched filtering, LFM
        pulse compression, and range-sidelobe reduction by weighting.
    """
    kwargs = window_kwargs or {}
    name = window.lower()
    if name == "none":
        win = np.ones(n_taps)
    elif name == "chebyshev":
        win = signal.windows.chebwin(n_taps, kwargs.get("at", 60.0))
    elif name == "blackman-harris":
        win = signal.windows.blackmanharris(n_taps)
    elif name == "taylor":
        win = signal.windows.taylor(n_taps, **kwargs)
    else:
        raise ValueError(
            f"Unknown window type '{window}'. "
            "Choose from: 'chebyshev', 'blackman-harris', 'taylor', 'none'."
        )
    return win / np.mean(win)


def matchfilter(
    datacube: np.ndarray,
    pulse_wvf: np.ndarray,
    pedantic: bool = True,
    window: str = "none",
    window_kwargs: dict | None = None,
) -> None:
    """Applies a matched filter to a datacube for pulse compression.

    Mirrors a real-time hardware matched filter (FIR with coefficients
    ``p*[-n]``): output is the raw discrete correlation with no additional
    scaling.  For a unit-amplitude transmit pulse of ``N_taps`` samples the
    peak output is ``V_rx · N_taps`` where ``N_taps = T · fs ≈ T · B`` is
    the pulse-compression (TB) gain.

    Two implementations are available:
    - Pedantic (True): Iteratively applies the matched filter to each pulse
      using a time-domain helper function. This is typically slower but can
      be clearer to understand.
    - Non-pedantic (False): Uses a more efficient frequency-domain approach
      by performing convolution via FFT. This involves a single FFT of the
      waveform kernel and is generally faster for large datacubes.

    Pass ``window`` to weight the replica for range-sidelobe control: this
    suppresses the ~-13.2 dB LFM sidelobes that a CFAR detector would otherwise
    flag as spurious targets around a strong return, at the cost of a broader
    mainlobe and a small SNR loss.  See :func:`range_window` and Richards,
    *FRSP* 2nd ed., Ch. 4.

    Args:
        datacube: 2D time-domain datacube with shape (N_range_bins, N_pulses),
            modified in-place.
        pulse_wvf: 1D transmitted pulse template (unit-amplitude convention,
            see :class:`rad_lab.waveform.WaveformSample`).
        pedantic: If True, use the iterative time-domain helper; if False,
            use FFT-based convolution.  Defaults to True.
        window: Range weighting applied to the replica.  ``"none"`` (default)
            reproduces the plain matched filter.  See :func:`range_window`
            for the other options.
        window_kwargs: Optional dict forwarded to the window function.

    Returns:
        None: The `datacube` is modified in-place.
    """
    replica = pulse_wvf
    if window.lower() != "none":
        replica = range_window(pulse_wvf.size, window, window_kwargs) * pulse_wvf

    if pedantic:
        for j in range(datacube.shape[1]):
            _, mf = matchfilter_with_waveform(datacube[:, j], replica)
            datacube[:, j] = mf
    else:
        kernel = np.conj(replica)[::-1]
        datacube[:] = signal.fftconvolve(datacube, kernel.reshape(-1, 1), mode="same", axes=0)
