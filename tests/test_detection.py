"""Detection-theory checks against closed forms and textbook values.

References: Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., Ch. 6.
"""

import numpy as np
import pytest
from rad_lab.detection import (
    threshold_factor,
    threshold_factor_nci,
    pd_swerling0,
    pd_swerling1,
    pd_swerling3,
    pd_swerling0_nci,
    required_snr,
    required_snr_nci,
    albersheim,
)

PFA = 1e-6


def test_threshold_factor_inverts_pfa():
    # Pfa = exp(-V_T) for a single square-law sample
    pfa = np.array([1e-3, 1e-6, 1e-8])
    assert np.exp(-threshold_factor(pfa)) == pytest.approx(pfa)


def test_threshold_factor_nci_single_pulse_matches_single_sample():
    # chi-squared with 2 DOF is the exponential distribution
    assert threshold_factor_nci(PFA, 1) == pytest.approx(threshold_factor(PFA))


def test_pd_equals_pfa_at_zero_snr():
    # with no signal present, a "detection" can only be a false alarm
    assert pd_swerling0(1e-12, 1e-3) == pytest.approx(1e-3, rel=1e-6)
    assert pd_swerling1(0.0, 1e-3) == pytest.approx(1e-3)


def test_pd_monotonic_in_snr():
    # non-decreasing everywhere (Pd saturates to 1.0 in float at high SNR),
    # strictly increasing below saturation
    snr = 10 ** (np.linspace(-5, 25, 61) / 10)
    for pd_func in (pd_swerling0, pd_swerling1, pd_swerling3):
        pd_vals = pd_func(snr, PFA)
        assert np.all(np.diff(pd_vals) >= 0)
        unsaturated = pd_vals[pd_vals < 0.999]
        assert np.all(np.diff(unsaturated) > 0)


def test_required_snr_swerling0_textbook_value():
    # Pd = 0.9, Pfa = 1e-6 requires ~13.2 dB for a non-fluctuating target
    assert required_snr(0.9, PFA) == pytest.approx(13.2, abs=0.1)


def test_required_snr_swerling1_closed_form():
    # invert Pd = Pfa^(1/(1+SNR)):  SNR = ln(Pfa)/ln(Pd) - 1
    expected_db = 10 * np.log10(np.log(PFA) / np.log(0.9) - 1)
    assert required_snr(0.9, PFA, "swerling1") == pytest.approx(expected_db, abs=1e-6)


def test_fluctuation_ordering_at_high_pd():
    # for high Pd, fluctuating targets need more SNR: Swerling I worst, III between
    snr0 = required_snr(0.9, PFA, "swerling0")
    snr3 = required_snr(0.9, PFA, "swerling3")
    snr1 = required_snr(0.9, PFA, "swerling1")
    assert snr0 < snr3 < snr1


def test_required_snr_round_trip():
    snr_db = required_snr(0.8, 1e-4)
    assert pd_swerling0(10 ** (snr_db / 10), 1e-4) == pytest.approx(0.8, abs=1e-9)


def test_nci_single_pulse_matches_pd_swerling0():
    snr = 10.0
    assert pd_swerling0_nci(snr, PFA, 1) == pytest.approx(pd_swerling0(snr, PFA))


def test_nci_improves_detection_at_fixed_per_pulse_snr():
    snr = 10 ** (5 / 10)
    assert pd_swerling0_nci(snr, PFA, 10) > pd_swerling0(snr, PFA)


def test_nci_integration_gain_bounds():
    # per-pulse SNR requirement drops with N, but by less than the coherent 10*log10(N)
    n_pulses = 16
    snr_1 = required_snr_nci(0.9, PFA, 1)
    snr_n = required_snr_nci(0.9, PFA, n_pulses)
    gain_db = snr_1 - snr_n
    assert 0 < gain_db < 10 * np.log10(n_pulses)


def test_albersheim_matches_exact_inverse():
    # quoted accuracy is ~0.3 dB in the region Pd=0.9, Pfa=1e-6
    for n_pulses in (1, 10, 64):
        exact_db = required_snr_nci(0.9, PFA, n_pulses)
        assert albersheim(0.9, PFA, n_pulses) == pytest.approx(exact_db, abs=0.5)
