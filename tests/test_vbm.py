"""VBM slow-time noise checks: Doppler spread sizing and spectral containment."""

import numpy as np
from numpy.linalg import norm
from rad_lab import constants as c
from rad_lab import vbm

N_PULSES = 256
PRF = 1e3
FCAR = 10e9
RDOT_DELTA = 3.0  # -> f_delta = 2*fcar/C * 3 = 200 Hz
F_DELTA = 200.0


def _inband_power_fraction(sequence):
    """Fraction of total power within the intended +/- f_delta/2 Doppler band.

    The band edges are widened by 10% to allow for spectral leakage.
    """
    freqs = np.fft.fftfreq(sequence.size, 1 / PRF)
    power = np.abs(np.fft.fft(sequence)) ** 2
    in_band = np.abs(freqs) <= 1.1 * F_DELTA / 2
    return power[in_band].sum() / power.sum()


def test_calc_f_delta():
    # Doppler spread of a range-rate spread: f_delta = 2*fcar/C * rdot_delta
    assert np.isclose(vbm.calc_f_delta(FCAR, RDOT_DELTA), 2 * FCAR / c.C * RDOT_DELTA)
    assert np.isclose(vbm.calc_f_delta(FCAR, RDOT_DELTA), F_DELTA)


def test_default_noise_is_lfm_phase():
    # the default (deterministic) technique is the slow-time LFM sweep
    from_wrapper = vbm.slowtime_noise(N_PULSES, FCAR, RDOT_DELTA, PRF)
    direct = vbm._lfm_phase(N_PULSES, F_DELTA, PRF)
    assert from_wrapper.shape == (N_PULSES,)
    assert np.allclose(from_wrapper, direct)
    # pure phase modulation: unit magnitude keeps the DRFM amplifier saturated
    assert np.allclose(np.abs(from_wrapper), 1.0)


def test_random_phase_is_unit_magnitude_broadband():
    np.random.seed(0)
    sequence = vbm._random_phase(N_PULSES)
    assert sequence.shape == (N_PULSES,)
    assert np.allclose(np.abs(sequence), 1.0)
    # broadband: energy spread over the full PRF, not confined to f_delta
    assert _inband_power_fraction(sequence) < 0.5


def test_band_limited_techniques_concentrate_power_in_band():
    np.random.seed(0)
    for noise_fun, min_fraction in [
        (vbm._lfm_phase, 0.95),
        (vbm._uniform_bandwidth_phase, 0.85),
        (vbm._gaussian_bandwidth_phase, 0.7),
        (vbm._gaussian_bandwidth_phase_normalized, 0.7),
    ]:
        sequence = vbm.slowtime_noise(N_PULSES, FCAR, RDOT_DELTA, PRF, noise_fun=noise_fun)
        assert sequence.shape == (N_PULSES,)
        assert _inband_power_fraction(sequence) > min_fraction, noise_fun.__name__


def test_gaussian_normalized_preserves_total_energy():
    np.random.seed(0)
    sequence = vbm._gaussian_bandwidth_phase_normalized(N_PULSES, F_DELTA, PRF)
    assert np.isclose(norm(sequence), np.sqrt(N_PULSES))
