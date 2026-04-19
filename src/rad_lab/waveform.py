"""Radar waveform definitions and factory functions.

Provides the :class:`WaveformSample` dataclass and factory functions for four
waveform types: uncoded (rectangular), Barker-coded, random phase-coded, and
linear frequency modulated (LFM) chirp.
"""

from enum import StrEnum
from dataclasses import dataclass, field
from typing import Callable
from functools import partial

import numpy as np
from numpy.linalg import norm
from . import constants as c


class WaveformType(StrEnum):
    """Enumeration of supported radar waveform types."""

    UNCODED = "uncoded"
    BARKER = "barker"
    RANDOM = "random"
    LFM = "lfm"


@dataclass
class WaveformSample:
    """Container for a radar waveform and its key parameters.

    Created by the factory functions (:func:`uncoded_waveform`,
    :func:`barker_coded_waveform`, :func:`random_coded_waveform`,
    :func:`lfm_waveform`). Call :meth:`set_sample` before passing to
    :func:`rad_lab.rdm.gen` to generate the discrete pulse array at the
    radar's sample rate.

    Attributes:
        type: Waveform type identifier.
        bw: Waveform bandwidth [Hz].
        time_bw_product: Time-bandwidth product [dimensionless]. Equal to 1
            for uncoded pulses; greater than 1 for coded waveforms, indicating
            pulse-compression gain.
        pulse_width: Pulse duration [s].
        pulse_func: Callable that generates ``(time_array, pulse_array)`` when
            called with a sample rate [Hz].
        pulse_sample: Discrete complex pulse array at the radar sample rate.
            Unit-amplitude (``|pulse_sample[n]| = 1`` over the pulse) — i.e.
            a physical 1-Volt transmit envelope, with no DSP normalisation
            applied.  The TB pulse-compression gain emerges from
            :func:`rad_lab.rf_datacube.matchfilter` (which integrates with
            ``Δt = 1/sample_rate``), not from rescaling stored here.
            Populated by :meth:`set_sample`; not set at construction time.
    """

    type: WaveformType
    bw: float
    time_bw_product: float
    pulse_width: float
    pulse_func: Callable
    pulse_sample: np.ndarray = field(init=False)

    def set_sample(self, sample_rate: float) -> None:
        """Generate and store the discrete pulse array at the given sample rate.

        Args:
            sample_rate: ADC sampling rate [Hz].
        """
        _, self.pulse_sample = self.pulse_func(sample_rate)


BARKER_DICT = {
    2: [1, -1],
    3: [1, 1, -1],
    4: [1, 1, -1, 1],
    5: [1, 1, 1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


def uncoded_pulse(
    sample_rate: float, bw: float, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a simple, baseband rectangular (uncoded) pulse.

    Args:
        sample_rate (float): The sampling rate in Hz.
        bw (float): The pulse bandwidth in Hz. The pulse duration is 1/bw.
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag (np.ndarray): The real-valued magnitude of the pulse samples.

    Raises:
        AssertionError: If the sample rate is below the Nyquist rate (2 * bw).
    """
    assert sample_rate / bw >= 2, "sample rate must be >= 2 * bw (Nyquist)"

    T = 1 / bw
    dt = 1 / sample_rate
    t = np.arange(0, T, dt)
    mag = np.ones(t.size)

    if normalize:
        mag = mag / norm(mag)

    return t, mag


def complex_tone_pulse(
    sample_rate: float, bw: float, fc: float, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a complex-valued pulse with a constant frequency offset.

    This function creates a rectangular pulse and modulates it with a complex
    exponential at a given carrier frequency.

    Args:
        sample_rate (float): The sampling rate in Hz.
        bw (float): The pulse bandwidth in Hz. The pulse duration is 1/bw.
        fc (float): The carrier frequency offset in Hz.
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag_c (np.ndarray): The complex-valued samples of the pulse.

    Raises:
        AssertionError: If the sample rate is below the Nyquist rate (2 * bw).
    """
    t, mag = uncoded_pulse(sample_rate, bw, normalize=normalize)
    mag_c = np.exp(2j * c.PI * fc * t) * mag
    return t, mag_c


def coded_pulse(
    sample_rate: float, bw: float, code: list[int], normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a baseband, phase-coded pulse.

    The pulse is constructed by concatenating rectangular "chips", where each
    chip has a phase determined by the input code (1 for 0 phase, -1 for pi
    phase).

    Args:
        sample_rate (float): The sampling rate in Hz.
        bw (float): The chip bandwidth in Hz. The chip duration is 1/bw.
        code (list[int]): A list of code values, which must be either 1 or -1.
            The length of the list determines the number of chips.
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag (np.ndarray): The real-valued, coded magnitude of the pulse.

    Raises:
        AssertionError: If any value in the code is not 1 or -1.
    """
    Tc = 1 / bw
    dt = 1 / sample_rate
    samples_per_chip = round(Tc * sample_rate)

    code_array = np.asarray(code)
    assert np.all(np.abs(code_array) == 1), "Code values must be either 1 or -1."
    mag = np.repeat(code_array, samples_per_chip).astype(float)
    t = np.arange(mag.size) * dt

    if normalize:
        mag = mag / norm(mag)

    return t, mag


def barker_coded_pulse(
    sample_rate: float, bw: float, nchips: int, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a baseband, Barker-coded pulse.

    Barker codes are specific binary phase codes known for their low
    autocorrelation sidelobes.

    Args:
        sample_rate (float): The sampling rate in Hz.
        bw (float): The chip bandwidth in Hz. The chip duration is 1/bw.
        nchips (int): The number of chips in the Barker code. Must be a valid
            Barker code length (2, 3, 4, 5, 7, 11, or 13).
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag (np.ndarray): The real-valued, Barker-coded magnitude of the pulse.

    Raises:
        AssertionError: If nChips is not a valid Barker code length.
    """
    assert nchips in BARKER_DICT, f"{nchips=} is not a valid Barker code length."
    return coded_pulse(
        sample_rate,
        bw,
        BARKER_DICT[nchips],
        normalize=normalize,
    )


def random_coded_pulse(
    sample_rate: float, bw: float, nchips: int, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a baseband pulse with a random binary phase code.

    The code consists of a sequence of randomly chosen 1s and -1s.

    Args:
        sample_rate (float): The sampling rate in Hz.
        bw (float): The chip bandwidth in Hz. The chip duration is 1/bw.
        nchips (int): The number of chips in the random code.
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag (np.ndarray): The real-valued, randomly coded magnitude.
    """
    code_rand = np.random.choice([1, -1], size=nchips)
    return coded_pulse(
        sample_rate,
        bw,
        code_rand,
        normalize=normalize,
    )


def lfm_pulse(
    sample_rate: float, bw: float, T: float, chirp_up_down: int, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a baseband Linear Frequency Modulated (LFM) pulse (chirp).

    The instantaneous frequency of the pulse varies linearly with time over the
    pulse duration.

    Args:
        sample_rate (float): The sampling rate in Hz.
        bw (float): The bandwidth of the frequency sweep in Hz.
        T (float): The total pulse duration in seconds.
        chirp_up_down (int): Determines the direction of the frequency sweep.
            1 for an up-chirp (frequency increases), -1 for a down-chirp
            (frequency decreases).
        normalize (bool, optional): If True, the pulse is normalized to have
            unit energy. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t (np.ndarray): Time vector for the pulse in seconds.
            - mag (np.ndarray): The complex-valued samples of the LFM pulse.

    Raises:
        AssertionError: If chirp_up_down is not 1 or -1.
    """
    assert chirp_up_down in [1, -1], "chirp_up_down must be either 1 or -1."
    dt = 1 / sample_rate
    t = np.arange(0, T, dt)
    k = bw / T  # Chirp rate
    phase = chirp_up_down * np.pi * k * (t**2) - chirp_up_down * np.pi * bw * t
    mag = np.exp(1j * phase)

    if normalize:
        mag = mag / norm(mag)

    return t, mag


# --- Waveform factories --------------------------------------------------------


def uncoded_waveform(bw: float) -> WaveformSample:
    """Returns a WaveformSample for an uncoded rectangular pulse.

    Args:
        bw: Pulse bandwidth in Hz. Pulse duration is 1/bw.

    Returns:
        WaveformSample for use with rdm.gen.
    """
    return WaveformSample(
        type=WaveformType.UNCODED,
        bw=bw,
        time_bw_product=1,
        pulse_width=1 / bw,
        pulse_func=partial(uncoded_pulse, bw=bw, normalize=False),
    )


def barker_coded_waveform(bw: float, nchips: int) -> WaveformSample:
    """Returns a WaveformSample for a Barker-coded pulse.

    Args:
        bw: Chip bandwidth in Hz. Chip duration is 1/bw.
        nchips: Number of chips. Must be a valid Barker length (2,3,4,5,7,11,13).

    Returns:
        WaveformSample for use with rdm.gen.
    """
    assert nchips in BARKER_DICT, f"nchips={nchips} is not a valid Barker code length."
    return WaveformSample(
        type=WaveformType.BARKER,
        bw=bw,
        time_bw_product=nchips,
        pulse_width=nchips / bw,
        pulse_func=partial(barker_coded_pulse, bw=bw, nchips=nchips, normalize=False),
    )


def random_coded_waveform(bw: float, nchips: int) -> WaveformSample:
    """Returns a WaveformSample for a random binary phase-coded pulse.

    Args:
        bw: Chip bandwidth in Hz. Chip duration is 1/bw.
        nchips: Number of chips.

    Returns:
        WaveformSample for use with rdm.gen.
    """
    return WaveformSample(
        type=WaveformType.RANDOM,
        bw=bw,
        time_bw_product=nchips,
        pulse_width=nchips / bw,
        pulse_func=partial(random_coded_pulse, bw=bw, nchips=nchips, normalize=False),
    )


def lfm_waveform(bw: float, T: float, chirp_up_down: int) -> WaveformSample:
    """Returns a WaveformSample for a Linear Frequency Modulated (LFM) pulse.

    Args:
        bw: Bandwidth of the frequency sweep in Hz.
        T: Pulse duration in seconds.
        chirp_up_down: 1 for up-chirp, -1 for down-chirp.

    Returns:
        WaveformSample for use with rdm.gen.
    """
    assert chirp_up_down in [1, -1], "chirp_up_down must be 1 (up) or -1 (down)."
    return WaveformSample(
        type=WaveformType.LFM,
        bw=bw,
        time_bw_product=bw * T,
        pulse_width=T,
        pulse_func=partial(lfm_pulse, bw=bw, T=T, chirp_up_down=chirp_up_down, normalize=False),
    )
