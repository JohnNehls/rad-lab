import numpy as np
import pytest
import rad_lab.pulse_doppler_radar as pdr
from rad_lab import rdm, Radar, Target, EaPlatform, Return
from rad_lab import uncoded_waveform, barker_coded_waveform, random_coded_waveform, lfm_waveform

BW = 10e6

RADAR = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=0,  # noiseless: argmax finds the target peak unambiguously
    sample_rate=2 * BW,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=50e3,
    dwell_time=2e-3,
)

RETURN = Return(
    target=Target(range=8.4e3, range_rate=2.0e3),
    platform=EaPlatform(
        tx_power=5.0e3,
        tx_gain=10 ** (30 / 10),
        total_losses=10 ** (3 / 10),
        rdot_delta=0.1e3,
        rdot_offset=0.0e3,
        range_offset=0.0e3,
        delay=0.0e-6,
    ),
)

WAVEFORMS = [
    uncoded_waveform(BW),
    barker_coded_waveform(BW, nchips=5),
    barker_coded_waveform(BW, nchips=13),
    random_coded_waveform(BW, nchips=13),
    lfm_waveform(BW, T=10 / 40e6, chirp_up_down=1),
]


def check_max_in_expected_bin(waveform):
    rdot_axis, r_axis, datacube = rdm.gen(RADAR, waveform, [RETURN], plot=False)

    range_expected = pdr.range_aliased(RETURN.target.range, RADAR.prf)
    rangeRate_expected = pdr.range_rate_aliased_prf_f0(
        RETURN.target.range_rate, RADAR.prf, RADAR.fcar
    )
    i = np.argmin(abs(r_axis - range_expected))
    j = np.argmin(abs(rdot_axis - rangeRate_expected))

    max_index_flat = np.argmax(abs(datacube))
    max_range_index, max_rdot_index = np.unravel_index(max_index_flat, datacube.shape)

    assert max_range_index == i, f"Range bin mismatch: got {max_range_index}, expected {i}"
    assert max_rdot_index == j, f"Doppler bin mismatch: got {max_rdot_index}, expected {j}"


@pytest.mark.parametrize("waveform", WAVEFORMS, ids=lambda w: f"{w.type}_{w.time_bw_product}")
def test_jammer_return_in_correct_bin(waveform):
    check_max_in_expected_bin(waveform)
