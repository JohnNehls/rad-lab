"""Range-equation checks: physical scaling laws and round-trip consistency."""

import pytest
from rad_lab import constants as c
from rad_lab.range_equation import (
    signal_range_eqn,
    signal_range_eqn_one_way,
    noise_power,
    snr_range_eqn_uncoded,
    snr_range_eqn,
    snr_range_eqn_cp,
    snr_range_eqn_bpsk_cp,
    snr_range_eqn_duty_factor_pulses,
    max_target_detection_range,
    max_target_detection_range_bpsk_cp,
    max_target_detection_range_dutyfactor_cp,
)

# X-band-ish reference point, all gains/losses linear
PT, GT, GR = 1e3, 1e3, 1e3
SIGMA, WAVELENGTH, R = 1.0, 0.03, 20e3
B, F, L, T = 1e6, 2.0, 2.0, 290.0


def test_signal_power_with_unity_parameters():
    # Pt = (4*pi)^3 with every other factor 1 gives exactly 1 W at R = 1 m
    assert signal_range_eqn((4 * c.PI) ** 3, 1, 1, 1, 1, 1, 1) == pytest.approx(1.0)


def test_noise_power_with_unity_parameters():
    # T = 1/k with B = F = 1 gives exactly 1 W
    assert noise_power(1, 1, 1 / c.K_BOLTZ) == pytest.approx(1.0)


def test_two_way_equals_one_way_with_reflection_spreading():
    # the target intercepts the incident wave and re-radiates: an extra
    # sigma / (4*pi*R^2) spreading factor relative to the one-way link
    one_way = signal_range_eqn_one_way(PT, GT, GR, WAVELENGTH, R, L)
    two_way = signal_range_eqn(PT, GT, GR, SIGMA, WAVELENGTH, R, L)
    assert two_way == pytest.approx(one_way * SIGMA / (4 * c.PI * R**2))


def test_snr_is_signal_power_over_noise_power():
    signal = signal_range_eqn(PT, GT, GR, SIGMA, WAVELENGTH, R, L)
    noise = noise_power(B, F, T)
    snr = snr_range_eqn_uncoded(PT, GT, GR, SIGMA, WAVELENGTH, R, B, F, L, T)
    assert snr == pytest.approx(signal / noise)


def test_two_way_snr_falls_as_fourth_power_of_range():
    snr_r = snr_range_eqn_uncoded(PT, GT, GR, SIGMA, WAVELENGTH, R, B, F, L, T)
    snr_2r = snr_range_eqn_uncoded(PT, GT, GR, SIGMA, WAVELENGTH, 2 * R, B, F, L, T)
    assert snr_2r == pytest.approx(snr_r / 16)


def test_one_way_power_falls_as_square_of_range():
    p_r = signal_range_eqn_one_way(PT, GT, GR, WAVELENGTH, R, L)
    p_2r = signal_range_eqn_one_way(PT, GT, GR, WAVELENGTH, 2 * R, L)
    assert p_2r == pytest.approx(p_r / 4)


def test_pulse_compression_and_coherent_gain():
    tb_product, n_pulses = 13, 32
    snr_uncoded = snr_range_eqn_uncoded(PT, GT, GR, SIGMA, WAVELENGTH, R, B, F, L, T)
    snr_pc = snr_range_eqn(PT, GT, GR, SIGMA, WAVELENGTH, R, B, F, L, T, tb_product)
    snr_cp = snr_range_eqn_cp(PT, GT, GR, SIGMA, WAVELENGTH, R, B, F, L, T, n_pulses, tb_product)
    snr_bpsk = snr_range_eqn_bpsk_cp(
        PT, GT, GR, SIGMA, WAVELENGTH, R, B, F, L, T, n_pulses, tb_product
    )
    assert snr_pc == pytest.approx(snr_uncoded * tb_product)
    assert snr_cp == pytest.approx(snr_uncoded * tb_product * n_pulses)
    assert snr_bpsk == pytest.approx(snr_cp)


def test_max_range_round_trip():
    snr_thresh = 10 ** (13 / 10)
    r_max = max_target_detection_range(PT, GT, GR, SIGMA, WAVELENGTH, snr_thresh, B, F, L, T)
    snr_at_r_max = snr_range_eqn_uncoded(PT, GT, GR, SIGMA, WAVELENGTH, r_max, B, F, L, T)
    assert snr_at_r_max == pytest.approx(snr_thresh)


def test_max_range_bpsk_cp_round_trip():
    snr_thresh = 10 ** (13 / 10)
    n_pulses, n_chips = 32, 13
    r_max = max_target_detection_range_bpsk_cp(
        PT, GT, GR, SIGMA, WAVELENGTH, snr_thresh, B, F, L, T, n_pulses, n_chips
    )
    snr_at_r_max = snr_range_eqn_bpsk_cp(
        PT, GT, GR, SIGMA, WAVELENGTH, r_max, B, F, L, T, n_pulses, n_chips
    )
    assert snr_at_r_max == pytest.approx(snr_thresh)


def test_duty_factor_form_matches_pulse_form():
    # N uncoded pulses of width tau (B = 1/tau, TB = 1) carry the same energy
    # as a duty-factor description with Tcpi = N/prf and tau_df = tau*prf
    prf, tau, n_pulses = 10e3, 5e-6, 64
    snr_pulses = snr_range_eqn_cp(PT, GT, GR, SIGMA, WAVELENGTH, R, 1 / tau, F, L, T, n_pulses, 1)
    snr_df = snr_range_eqn_duty_factor_pulses(
        PT, GT, GR, SIGMA, WAVELENGTH, R, F, L, T, Tcpi=n_pulses / prf, tau_df=tau * prf
    )
    assert snr_df == pytest.approx(snr_pulses)


def test_duty_factor_out_of_range_raises():
    with pytest.raises(AssertionError):
        snr_range_eqn_duty_factor_pulses(
            PT, GT, GR, SIGMA, WAVELENGTH, R, F, L, T, Tcpi=1e-3, tau_df=1.5
        )


def test_max_range_duty_factor_round_trip():
    snr_thresh = 10 ** (13 / 10)
    tcpi, tau_df = 2e-3, 0.1
    r_max = max_target_detection_range_dutyfactor_cp(
        PT, GT, GR, SIGMA, WAVELENGTH, snr_thresh, F, L, T, tcpi, tau_df
    )
    snr_at_r_max = snr_range_eqn_duty_factor_pulses(
        PT, GT, GR, SIGMA, WAVELENGTH, r_max, F, L, T, tcpi, tau_df
    )
    assert snr_at_r_max == pytest.approx(snr_thresh)
