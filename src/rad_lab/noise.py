"""Complex Gaussian noise generation.

Provides unity-variance complex noise arrays and thermal noise power
calculations used to populate the receiver noise floor in the datacube.
"""

import numpy as np
from scipy import fft
from . import constants as c


def _unit_magnitude(noise: np.ndarray) -> np.ndarray:
    """Normalizes a complex signal to pointwise magnitude one (zeros stay zero)."""
    magnitude = np.abs(noise)
    return np.divide(
        noise,
        magnitude,
        out=np.zeros_like(noise, dtype=np.complex64),
        where=(magnitude != 0),
    )


def unity_variance_complex_noise(in_size: tuple | int) -> np.ndarray:
    """
    Generates complex Gaussian noise with unity variance.

    Args:
        in_size: Shape of the output array.

    Returns:
        A complex array where the real and imaginary components are
        independent standard normal distributions, scaled to achieve unit variance.
    """
    real_part = np.random.standard_normal(size=in_size)
    imag_part = np.random.standard_normal(size=in_size)
    return (real_part + 1j * imag_part) / np.sqrt(2)


def band_limited_complex_noise(
    f_min: float, f_max: float, N_samples: int, fs: float, normalize: bool = False
) -> np.ndarray:
    """
    Generates band-limited complex noise within a specified frequency range.

    Args:
        f_min: Minimum frequency limit in Hz.
        f_max: Maximum frequency limit in Hz.
        N_samples: Total number of time-domain samples to generate.
        fs: Sampling frequency in Hz.
        normalize: If True, normalizes the output signal to have a pointwise
            magnitude of one. Defaults to False.

    Returns:
        A complex time-domain array representing the band-limited noise.
    """
    if not isinstance(N_samples, int):
        raise TypeError("N_samples must be an integer.")
    if f_min > f_max:
        raise ValueError("f_min must not be greater than f_max.")
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")

    freqs = fft.fftfreq(N_samples, d=1 / fs)
    spectrum = np.zeros(N_samples, dtype=np.complex64)

    # Create a boolean mask for frequencies within the specified band
    frequency_band_mask = (freqs >= f_min) & (freqs <= f_max)

    # For frequencies in the band, create phasors (complex numbers with magnitude 1)
    num_freqs_in_band = np.sum(frequency_band_mask)
    random_phases = 2 * np.pi * np.random.rand(num_freqs_in_band)
    spectrum[frequency_band_mask] = np.exp(1j * random_phases)

    # Inverse FFT to get the time-domain signal
    noise = fft.ifft(spectrum)

    if normalize:
        return _unit_magnitude(noise)

    return noise


def gaussian_complex_noise(
    mu: float, sigma: float, p: float, N_samples: int, fs: float, normalize: bool = False
) -> np.ndarray:
    """
    Generates complex noise with a Power Spectral Density (PSD) shaped by a
    generalized Gaussian envelope.

    Args:
        mu: Center frequency of the Gaussian PSD in Hz.
        sigma: Standard deviation of the Gaussian distribution.
        p: Order of the Gaussian distribution (p=1 for standard Gaussian).
        N_samples: Total number of samples to generate.
        fs: Sampling frequency in Hz.
        normalize: If True, normalizes the output signal to have a pointwise
            magnitude of one. Defaults to False.

    Returns:
        A complex time-domain array with Gaussian-shaped spectral content.
    """
    if not isinstance(N_samples, int):
        raise TypeError("N_samples must be an integer.")
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")

    freqs = fft.fftfreq(N_samples, d=1 / fs)

    # Calculate the magnitude of the spectrum based on a generalized Gaussian shape
    gaussian_term = ((freqs - mu) ** 2) / (2 * sigma**2)
    magnitude_spectrum = np.exp(-(gaussian_term**p))

    # The original included a scaling factor from the normal distribution PDF.
    # While not strictly necessary for shaping, we keep it for consistency.
    pdf_scaling_factor = 1 / (sigma * np.sqrt(2 * c.PI))
    magnitude_spectrum *= pdf_scaling_factor

    # Apply a random phase to each frequency component to create the complex spectrum
    random_phases = 2 * c.PI * np.random.rand(N_samples)
    spectrum = magnitude_spectrum * np.exp(1j * random_phases)
    spectrum = spectrum.astype(np.complex64)

    # Inverse FFT to get the time-domain signal.
    # Scaling by sqrt(N_samples) helps preserve power according to Parseval's theorem.
    noise = fft.ifft(spectrum) * np.sqrt(N_samples)

    if normalize:
        return _unit_magnitude(noise)

    return noise
