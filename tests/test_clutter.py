import numpy as np
import pytest
import rad_lab.pulse_doppler_radar as pdr
from rad_lab import rdm, Radar, Clutter, uncoded_waveform
from rad_lab._rdm_internals import clutter_cell_rcs, gaussian_spectrum_sequence

BW = 10e6
PRF = 20e3

RADAR = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=0,  # noiseless: only clutter in the RDM
    sample_rate=2 * BW,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=PRF,
    dwell_time=3.2e-3,  # 64 pulses
)

CLUTTER = Clutter(
    range_start=2e3,
    range_stop=4e3,
    gamma=10 ** (-20 / 10),
    grazing_angle=np.deg2rad(2),
    azimuth_beamwidth=np.deg2rad(2),
    range_rate_std=0.1,
)


def test_cell_rcs_is_constant_gamma():
    dR = 7.5
    R = 3e3
    psi = CLUTTER.grazing_angle
    expected = CLUTTER.gamma * np.sin(psi) * R * CLUTTER.azimuth_beamwidth * dR / np.cos(psi)
    assert clutter_cell_rcs(CLUTTER, R, dR) == pytest.approx(expected)
    # Cell area, and so RCS, grows linearly with range.
    assert clutter_cell_rcs(CLUTTER, 2 * R, dR) == pytest.approx(2 * expected)


def test_gaussian_spectrum_sequence_has_unit_power():
    np.random.seed(0)
    seqs = np.array([gaussian_spectrum_sequence(64, PRF, 500.0) for _ in range(2000)])
    assert np.mean(np.abs(seqs) ** 2) == pytest.approx(1.0, rel=0.05)


def test_gaussian_spectrum_sequence_spectral_width():
    np.random.seed(0)
    n, doppler_std = 256, 1e3
    psd = np.zeros(n)
    for _ in range(500):
        psd += np.abs(np.fft.fft(gaussian_spectrum_sequence(n, PRF, doppler_std))) ** 2
    freqs = np.fft.fftfreq(n, d=1 / PRF)
    measured_std = np.sqrt(np.sum(psd * freqs**2) / np.sum(psd))
    assert measured_std == pytest.approx(doppler_std, rel=0.05)


def test_zero_spread_sequence_is_constant():
    np.random.seed(0)
    seq = gaussian_spectrum_sequence(64, PRF, 0.0)
    assert np.allclose(seq, seq[0])


@pytest.mark.parametrize("range_rate_mean", [0.0, 30.0])
def test_clutter_peaks_at_mean_doppler(range_rate_mean):
    clutter = Clutter(**{**CLUTTER.__dict__, "range_rate_mean": range_rate_mean})
    rdot_axis, _, datacube = rdm.gen(RADAR, uncoded_waveform(BW), [clutter], plot=False)

    doppler_power = np.sum(np.abs(datacube) ** 2, axis=0)
    expected = pdr.range_rate_aliased_prf_f0(range_rate_mean, RADAR.prf, RADAR.fcar)
    assert np.argmax(doppler_power) == np.argmin(abs(rdot_axis - expected))


def test_clutter_confined_to_range_band():
    _, r_axis, datacube = rdm.gen(RADAR, uncoded_waveform(BW), [CLUTTER], plot=False)

    range_power = np.sum(np.abs(datacube) ** 2, axis=1)
    margin = 10 * (r_axis[1] - r_axis[0])  # well beyond the pulse-compressed width
    inside = (r_axis > CLUTTER.range_start) & (r_axis < CLUTTER.range_stop)
    outside = (r_axis < CLUTTER.range_start - margin) | (r_axis > CLUTTER.range_stop + margin)
    # float32 datacube: leakage outside the band is only round-off.
    assert range_power[outside].max() < 1e-9 * range_power[inside].min()
