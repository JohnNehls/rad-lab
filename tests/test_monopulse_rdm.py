import numpy as np
import rad_lab.uniform_linear_arrays as ula
import rad_lab.monopulse as mp
from rad_lab import rdm, Target, Return, lfm_waveform
from _radars import x_band_radar

BW = 10e6

RADAR = x_band_radar(50e3, 2e-3)

WAVEFORM = lfm_waveform(BW, T=10 / 40e6, chirp_up_down=1)
TGT_ANGLE = 5  # degrees
DX = 1 / 2  # element separation in wavelengths


def test_monopulse_rdm_angle_error_within_threshold():
    array_pos = np.array([-DX / 2, DX / 2])
    steer_vec = ula.steering_vector(array_pos, TGT_ANGLE)

    dc_list = []
    for i, sv in enumerate(steer_vec):
        return_list = [Return(target=Target(range=2.4e3, range_rate=0.2e3, rcs=10, sv=sv))]
        _, _, datacube = rdm.gen(RADAR, WAVEFORM, return_list, debug=False, plot=False, seed=i)
        dc_list.append(datacube)

    measured_theta = mp.monopulse_angle_at_peak_deg(dc_list[0], dc_list[1], DX)
    error = abs(measured_theta - TGT_ANGLE)

    assert error < 1.0
