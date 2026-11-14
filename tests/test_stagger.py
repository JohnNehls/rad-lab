import dataclasses
import numpy as np
import pytest
from rad_lab import rdm, mti, Radar, Target, Return, EaPlatform, lfm_waveform
from rad_lab import pulse_doppler_radar as pdr
from rad_lab._rdm_internals import gaussian_spectrum_factor

BW = 10e6
T = 1e-6
STAGGER = (23, 27, 24, 26)  # base interval 40 samples at 20 MHz / 20 kHz

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
    pri_stagger=STAGGER,
)


def test_stagger_base_must_be_whole_samples():
    with pytest.raises(ValueError, match="whole number"):
        dataclasses.replace(RADAR, pri_stagger=(4, 5))  # 1000 / 4.5 samples


def test_pulse_tx_times_cycle_the_stagger():
    assert np.array_equal(RADAR.stagger_pri_samples, np.array(STAGGER) * 40)
    pris = np.diff(RADAR.pulse_tx_times) * RADAR.sample_rate
    assert np.allclose(pris, np.resize(np.array(STAGGER) * 40, RADAR.n_pulses - 1))
    uniform = dataclasses.replace(RADAR, pri_stagger=None)
    assert np.array_equal(uniform.pulse_tx_times, np.arange(uniform.n_pulses) / uniform.prf)


def test_gen_rejects_stagger():
    with pytest.raises(ValueError, match="gen_rtm"):
        rdm.gen(RADAR, lfm_waveform(BW, T, 1), [], plot=False)


def test_jammer_rejects_stagger():
    jammer = Return(
        target=Target(range=3e3, range_rate=0, rcs=None),
        platform=EaPlatform(tx_power=1, tx_gain=1, total_losses=1),
    )
    with pytest.raises(NotImplementedError):
        rdm.gen_rtm(RADAR, lfm_waveform(BW, T, 1), [jammer])


def test_staggered_target_in_correct_bin_every_pulse():
    quiet = dataclasses.replace(RADAR, op_temp=0)
    target = Target(range=3e3, range_rate=100, rcs=1)
    r_axis, rtm = rdm.gen_rtm(quiet, lfm_waveform(BW, T, 1), [Return(target=target)])
    assert len(r_axis) == min(STAGGER) * 40  # range gate is the shortest PRI
    i_r = np.argmin(abs(r_axis - target.range))
    assert np.all(np.argmax(np.abs(rtm), axis=0) == i_r)


def test_staggered_canceller_removes_stationary_target():
    quiet = dataclasses.replace(RADAR, op_temp=0)
    target = Return(target=Target(range=3e3, range_rate=0, rcs=1))
    _, rtm = rdm.gen_rtm(quiet, lfm_waveform(BW, T, 1), [target], mti_order=1)
    _, rtm_0 = rdm.gen_rtm(quiet, lfm_waveform(BW, T, 1), [target])
    assert np.max(np.abs(rtm)) < 1e-4 * np.max(np.abs(rtm_0))


def test_equal_ratios_match_constant_pri_response():
    freqs = np.linspace(-30e3, 30e3, 101)
    for order in [1, 2]:
        assert np.allclose(
            mti.frequency_response(order, 20e3, freqs, pri_stagger=(3, 3)),
            mti.frequency_response(order, 20e3, freqs),
        )


def test_stagger_blind_speed():
    blind = mti.blind_speeds(20e3, 10e9, n=2, pri_stagger=STAGGER)
    assert np.allclose(blind, [7500, 15000])  # 25x the 300 m/s constant-PRI blind speed
    doppler = pdr.frequency_delta_doppler(np.array([300.0, blind[0]]), 10e9)
    response = mti.frequency_response(1, 20e3, doppler, pri_stagger=STAGGER)
    assert response[0] > 0.1  # old blind speed is passed
    assert response[1] < 1e-20  # true blind speed is cancelled


def test_gaussian_spectrum_factor_covariance():
    times = RADAR.pulse_tx_times
    doppler_std = 500.0
    factor = gaussian_spectrum_factor(times, doppler_std)
    tau = times[:, None] - times[None, :]
    expected = np.exp(-2 * np.pi**2 * doppler_std**2 * tau**2)
    assert np.allclose(factor @ factor.conj().T, expected, atol=1e-10)
    stationary = gaussian_spectrum_factor(times, 0.0) @ np.ones(len(times))
    assert np.allclose(stationary, stationary[0])  # one amplitude for every pulse


def test_doppler_bank_matches_fft_for_constant_pri():
    uniform = dataclasses.replace(RADAR, pri_stagger=None)
    rng = np.random.default_rng(0)
    rtm = rng.standard_normal((4, uniform.n_pulses)) + 1j * rng.standard_normal((
        4,
        uniform.n_pulses,
    ))
    freqs = np.fft.fftfreq(uniform.n_pulses, 1 / uniform.prf)
    bank = rdm.doppler_bank(rtm, uniform.pulse_tx_times, freqs)
    assert np.allclose(bank, np.fft.fft(rtm, axis=1))


def test_doppler_bank_full_coherent_gain_with_stagger():
    quiet = dataclasses.replace(RADAR, op_temp=0)
    target = Target(range=3e3, range_rate=120, rcs=1)
    waveform = lfm_waveform(BW, T, 1)
    r_axis, rtm = rdm.gen_rtm(quiet, waveform, [Return(target=target)])
    i_r = np.argmin(abs(r_axis - target.range))
    single = np.abs(rtm[i_r]) ** 2
    doppler = np.array([pdr.frequency_delta_doppler(target.range_rate, RADAR.fcar)])
    bank = rdm.doppler_bank(rtm[i_r : i_r + 1], quiet.pulse_tx_times, doppler)
    assert np.abs(bank[0, 0]) ** 2 == pytest.approx(RADAR.n_pulses**2 * single.mean(), rel=1e-3)
