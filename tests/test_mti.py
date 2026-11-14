import numpy as np
import pytest
from rad_lab import rdm, mti, Radar, Target, Return, Clutter, uncoded_waveform
from rad_lab import constants as c
from rad_lab import pulse_doppler_radar as pdr
from rad_lab.noise import gaussian_spectrum_factor, unity_variance_complex_noise

BW = 10e6
PRF = 20e3
FCAR = 10e9

RADAR = Radar(
    fcar=FCAR,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * BW,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=PRF,
    dwell_time=3.2e-3,  # 64 pulses
)


@pytest.mark.parametrize(
    "order, expected", [(0, [1]), (1, [1, -1]), (2, [1, -2, 1]), (3, [1, -3, 3, -1])]
)
def test_canceller_coefficients(order, expected):
    assert np.array_equal(mti.canceller_coefficients(order), expected)


@pytest.mark.parametrize("order, expected", [(0, 1), (1, 2), (2, 6), (3, 20)])
def test_noise_gain(order, expected):
    assert mti.noise_gain(order) == expected


def test_order_one_canceller_is_pulse_difference():
    rng = np.random.default_rng(0)
    cube = rng.standard_normal((5, 16)) + 1j * rng.standard_normal((5, 16))
    assert np.allclose(mti.apply_canceller(cube, 1), np.diff(cube, axis=1))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_canceller_removes_stationary_return(order):
    cube = np.tile(np.arange(1, 6)[:, None] * (1 + 2j), (1, 16))
    filtered = mti.apply_canceller(cube, order)
    assert filtered.shape == (5, 16 - order)
    assert np.allclose(filtered, 0)


@pytest.mark.parametrize("order", [1, 2])
def test_frequency_response_closed_form(order):
    freqs = np.linspace(-PRF, PRF, 101)
    expected = (2 * np.sin(np.pi * freqs / PRF)) ** (2 * order)
    assert np.allclose(mti.frequency_response(order, PRF, freqs), expected)


def test_blind_speeds_are_multiples_of_half_wavelength_times_prf():
    wavelength = c.C / FCAR
    assert np.allclose(
        mti.blind_speeds(PRF, FCAR, n=3), [1, 2, 3] * np.array(wavelength * PRF / 2)
    )
    blind_doppler = pdr.frequency_delta_doppler(mti.blind_speeds(PRF, FCAR, n=3), FCAR)
    assert np.allclose(mti.frequency_response(2, PRF, blind_doppler), 0, atol=1e-20)


def test_order_one_improvement_factor_closed_form():
    range_rate_std = 1.0
    doppler_std = 2 * range_rate_std * FCAR / c.C
    rho = np.exp(-2 * np.pi**2 * doppler_std**2 / PRF**2)
    assert mti.improvement_factor(1, PRF, FCAR, range_rate_std) == pytest.approx(1 / (1 - rho))


def test_improvement_factor_finite_and_ordered_for_narrow_clutter():
    # A tap-correlation sum cancels to zero here; the spectral integral must not.
    factors = [mti.improvement_factor(order, PRF, FCAR, 0.01) for order in [1, 2, 3]]
    assert np.all(np.isfinite(factors))
    assert factors[0] < factors[1] < factors[2]
    assert mti.improvement_factor(1, PRF, FCAR, 0.0) == np.inf


@pytest.mark.parametrize("order", [1, 2])
def test_improvement_factor_matches_simulated_clutter(order):
    np.random.seed(0)
    range_rate_std = 3.0
    doppler_std = 2 * range_rate_std * FCAR / c.C
    factor = gaussian_spectrum_factor(np.arange(256) / PRF, doppler_std)
    clutter = (factor @ unity_variance_complex_noise((256, 400))).T

    clutter_gain = np.mean(np.abs(mti.apply_canceller(clutter, order)) ** 2) / np.mean(
        np.abs(clutter) ** 2
    )
    measured = mti.noise_gain(order) / clutter_gain
    expected = mti.improvement_factor(order, PRF, FCAR, range_rate_std)
    assert 10 * np.log10(measured) == pytest.approx(10 * np.log10(expected), abs=0.5)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_to_snr_noise_floor_is_unity_after_mti(order):
    waveform = uncoded_waveform(BW)
    _, _, datacube = rdm.gen(RADAR, waveform, [], plot=False, window="none", mti_order=order)
    snr_dc = rdm.to_snr(datacube, RADAR, waveform, mti_order=order)
    assert datacube.shape[1] == RADAR.n_pulses - order
    assert np.mean(np.abs(snr_dc) ** 2) == pytest.approx(1.0, rel=0.05)


def test_mti_suppresses_clutter_but_keeps_moving_target():
    waveform = uncoded_waveform(BW)
    target = Target(range=3e3, range_rate=60, rcs=1)
    clutter = Clutter(
        range_start=1e3,
        range_stop=6e3,
        gamma=10 ** (-20 / 10),
        grazing_angle=np.deg2rad(2),
        azimuth_beamwidth=np.deg2rad(2),
        range_rate_std=0.1,
    )
    return_list = [Return(target=target), clutter]

    results = {}
    for order in [0, 2]:
        rdot_axis, r_axis, datacube = rdm.gen(
            RADAR, waveform, return_list, plot=False, mti_order=order
        )
        snr_dc = np.abs(rdm.to_snr(datacube, RADAR, waveform, mti_order=order))
        i_r = np.argmin(abs(r_axis - target.range))
        i_v = np.argmin(abs(rdot_axis - target.range_rate))
        i_0 = np.argmin(abs(rdot_axis))
        results[order] = (snr_dc[:, i_0].max(), np.unravel_index(snr_dc.argmax(), snr_dc.shape))
        target_bin = (i_r, i_v)

    clutter_before, _ = results[0]
    clutter_after, peak_after = results[2]
    assert 20 * np.log10(clutter_before / clutter_after) > 30
    assert peak_after == target_bin  # with clutter gone, the target is the brightest cell
