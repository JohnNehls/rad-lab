import dataclasses
import numpy as np
import pytest
from rad_lab import rdm, Radar, Target, Return, lfm_waveform
from rad_lab import constants as c
from rad_lab.range_equation import snr_range_eqn
from rad_lab.rf_datacube import doppler_process

BW = 10e6
T = 1e-6

RADAR = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * BW,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=20e3,
    dwell_time=1.6e-3,  # 32 pulses
)

TARGET = Target(range=3e3, range_rate=40, rcs=1)  # 3 km is bin 399 (7.5 m bins, 1-indexed axis)


@pytest.mark.parametrize("mti_order", [0, 1])
def test_gen_equals_doppler_processed_rtm(mti_order):
    returns = [Return(target=TARGET)]
    _, r_axis, rdm_cube = rdm.gen(
        RADAR, lfm_waveform(BW, T, 1), returns, plot=False, window="none", mti_order=mti_order
    )
    r_axis_rtm, rtm = rdm.gen_rtm(RADAR, lfm_waveform(BW, T, 1), returns, mti_order=mti_order)
    doppler_process(rtm, RADAR.sample_rate)
    assert np.array_equal(r_axis, r_axis_rtm)
    assert np.array_equal(rdm_cube, rtm)


def test_nci_profile_is_rdm_power_over_doppler():
    """Parseval: sum_n |x[n]|^2 = sum_k |X[k]|^2 / N for the unwindowed FFT."""
    returns = [Return(target=TARGET)]
    _, _, rdm_cube = rdm.gen(RADAR, lfm_waveform(BW, T, 1), returns, plot=False, window="none")
    _, rtm = rdm.gen_rtm(RADAR, lfm_waveform(BW, T, 1), returns)
    from_rdm = np.sum(np.abs(rdm_cube) ** 2, axis=1) / RADAR.n_pulses
    assert np.allclose(rdm.noncoherent_integrate(rtm), from_rdm, rtol=1e-4)


@pytest.mark.parametrize("mti_order", [0, 2])
def test_noise_only_profile_mean_is_n_pulses(mti_order):
    waveform = lfm_waveform(BW, T, 1)
    _, rtm = rdm.gen_rtm(RADAR, waveform, [], mti_order=mti_order)
    profile = rdm.noncoherent_integrate(rdm.rtm_to_snr(rtm, RADAR, waveform, mti_order))
    assert np.mean(profile) == pytest.approx(RADAR.n_pulses - mti_order, rel=0.03)


def test_rtm_peak_is_single_pulse_snr():
    waveform = lfm_waveform(BW, T, 1)
    quiet_radar = dataclasses.replace(RADAR, op_temp=0)
    r_axis, rtm = rdm.gen_rtm(quiet_radar, waveform, [Return(target=TARGET)])
    snr_rtm = rdm.rtm_to_snr(rtm, RADAR, waveform)  # normalise by the noisy radar's noise
    expected = snr_range_eqn(
        RADAR.tx_power,
        RADAR.tx_gain,
        RADAR.rx_gain,
        TARGET.rcs,
        c.C / RADAR.fcar,
        TARGET.range,
        BW,
        RADAR.noise_factor,
        RADAR.total_losses,
        RADAR.op_temp,
        BW * T,
    )
    i_r = np.argmin(abs(r_axis - TARGET.range))
    assert np.allclose(np.abs(snr_rtm[i_r]) ** 2, expected, rtol=1e-3)
    profile = rdm.noncoherent_integrate(snr_rtm)
    assert profile[i_r] == pytest.approx(RADAR.n_pulses * expected, rel=1e-3)
