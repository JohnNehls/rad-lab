"""Ambiguity-function checks: axes, peak, symmetry, resolution, LFM coupling."""

import numpy as np
import pytest
from rad_lab.ambiguity import ambiguity_function
from rad_lab.waveform import uncoded_pulse, lfm_pulse

FS = 20e6
BW = 5e6
T = 10e-6  # LFM time-bandwidth product = 50
FD_MAX = 2e6
N_FD = 81


def _lfm_surface(chirp_up_down=1):
    _, pulse = lfm_pulse(FS, BW, T, chirp_up_down)
    return ambiguity_function(pulse, FS, FD_MAX, N_FD)


def _uncoded_surface():
    # uncoded pulse with the same duration T as the LFM (bw = 1/T)
    _, pulse = uncoded_pulse(FS, 1 / T)
    return ambiguity_function(pulse, FS, FD_MAX, N_FD)


def test_axes_shapes_and_spans():
    _, pulse = lfm_pulse(FS, BW, T, 1)
    tau, fd, amb = ambiguity_function(pulse, FS, FD_MAX, N_FD)
    n = len(pulse)
    assert tau.shape == (2 * n - 1,)
    assert np.allclose(tau, -tau[::-1])  # delay axis symmetric about zero
    assert fd.shape == (N_FD,)
    assert fd[0] == -FD_MAX and fd[-1] == FD_MAX
    assert amb.shape == (N_FD, 2 * n - 1)


def test_peak_is_one_at_origin():
    tau, fd, amb = _lfm_surface()
    i_fd, i_tau = np.unravel_index(np.argmax(amb), amb.shape)
    assert amb[i_fd, i_tau] == 1.0
    assert fd[i_fd] == 0.0
    assert tau[i_tau] == 0.0


def test_ambiguity_symmetry():
    # |chi(-tau, -fd)| = |chi(tau, fd)| for any waveform
    _, _, amb = _lfm_surface()
    assert np.allclose(amb, amb[::-1, ::-1], atol=1e-12)


def test_uncoded_zero_delay_cut_null_at_inverse_pulse_width():
    # zero-delay cut of a rectangular pulse is |sinc(fd*T)|^2: first null at fd = 1/T
    tau, fd, amb = _uncoded_surface()
    zero_delay_cut = amb[:, np.argmin(np.abs(tau))]
    i_null = np.argmin(np.abs(fd - 1 / T))
    assert zero_delay_cut[i_null] < 1e-3


def test_lfm_compression_narrows_zero_doppler_mainlobe():
    # same pulse duration, but the LFM mainlobe is ~TB times narrower
    _, fd_l, amb_lfm = _lfm_surface()
    _, fd_u, amb_unc = _uncoded_surface()
    lfm_halfwidth = np.sum(amb_lfm[np.argmin(np.abs(fd_l)), :] > 0.5)
    uncoded_halfwidth = np.sum(amb_unc[np.argmin(np.abs(fd_u)), :] > 0.5)
    assert lfm_halfwidth < uncoded_halfwidth / 10


def test_lfm_range_doppler_coupling():
    # a Doppler shift fd displaces the LFM correlation peak by tau = -fd/k
    # (up-chirp, k = bw/T); the down-chirp shifts the opposite way
    fd_0 = 0.5e6
    chirp_rate = BW / T
    expected_shift = fd_0 / chirp_rate  # 1 us
    one_sample = 1 / FS

    tau, fd, amb_up = _lfm_surface(chirp_up_down=1)
    _, _, amb_down = _lfm_surface(chirp_up_down=-1)
    i_fd = np.argmin(np.abs(fd - fd_0))
    peak_tau_up = tau[np.argmax(amb_up[i_fd, :])]
    peak_tau_down = tau[np.argmax(amb_down[i_fd, :])]
    assert peak_tau_up == pytest.approx(-expected_shift, abs=2 * one_sample)
    assert peak_tau_down == pytest.approx(+expected_shift, abs=2 * one_sample)


def test_uncoded_peak_does_not_shift_with_doppler():
    # no range-Doppler coupling for an unmodulated pulse
    tau, fd, amb = _uncoded_surface()
    i_fd = np.argmin(np.abs(fd - 0.5 / T))  # within the mainlobe Doppler extent
    assert tau[np.argmax(amb[i_fd, :])] == pytest.approx(0.0, abs=2 / FS)
