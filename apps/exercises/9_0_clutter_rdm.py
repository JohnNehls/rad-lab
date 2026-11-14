#!/usr/bin/env python
"""Surface clutter in the range-Doppler map.

Add a band of constant-gamma ground clutter to an RDM containing one slow
target and sweep the clutter's internal-motion spread (``range_rate_std``).

Notes:
  - Clutter lives at (near) zero Doppler across every range it covers; the
    target is visible only because its Doppler separates it from the ridge.
  - The clutter-to-noise ratio (CNR) falls slowly with range: cell area grows
    as R, so clutter power goes as R^-3 instead of a point target's R^-4.
  - Increasing ``range_rate_std`` widens the ridge in Doppler.  Wind-blown
    ground is ~0.1 m/s; sea and rain can reach several m/s (10 m/s is an
    extreme case).  A wide ridge swallows slow targets, which is what MTI and
    Doppler filtering must fight.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm, Radar, Target, Return, Clutter, lfm_waveform

bw = 10e6  # waveform bandwidth [Hz]

# -- Radar: low PRF so the clutter band is range-unambiguous (Ru = 7.5 km) --
radar = Radar(
    fcar=10e9,  # carrier frequency [Hz]
    tx_power=1e3,  # transmit power [W]
    tx_gain=10 ** (30 / 10),  # transmit antenna gain [linear], 30 dB
    rx_gain=10 ** (30 / 10),  # receive antenna gain [linear], 30 dB
    op_temp=290,  # operating temperature [K]
    sample_rate=2 * bw,  # Nyquist sampling of the waveform bandwidth
    noise_factor=10 ** (8 / 10),  # receiver noise factor [linear], 8 dB
    total_losses=10 ** (8 / 10),  # total system losses [linear], 8 dB
    prf=20e3,  # pulse repetition frequency [Hz]
    dwell_time=5e-3,  # coherent processing interval [s], 100 pulses
)

waveform = lfm_waveform(bw, T=1.0e-6, chirp_up_down=1)  # 1 us up-chirp

# -- Target: 1 m^2 at 4 km, receding at 40 m/s --
target = Target(range=4e3, range_rate=40, rcs=1)

for range_rate_std in [0.1, 3.0, 10.0]:
    # -- Farmland-like clutter from 1 to 6 km --
    clutter = Clutter(
        range_start=1e3,
        range_stop=6e3,
        gamma=10 ** (-20 / 10),  # -20 dB constant gamma
        grazing_angle=np.deg2rad(2),
        azimuth_beamwidth=np.deg2rad(2),
        range_rate_std=range_rate_std,  # internal-motion spread [m/s]
    )

    return_list = [Return(target=target), clutter]
    rdot_axis, r_axis, datacube = rdm.gen(radar, waveform, return_list, plot=False)
    snr_dc = rdm.to_snr(datacube, radar, waveform)

    # SNR at the target cell and peak clutter-to-noise ratio on the zero-Doppler ridge
    i_r = np.argmin(abs(r_axis - target.range))
    i_v = np.argmin(abs(rdot_axis - target.range_rate))
    i_0 = np.argmin(abs(rdot_axis))
    print(f"range_rate_std = {range_rate_std:.1f} m/s")
    print(f"  target SNR      = {20 * np.log10(abs(snr_dc[i_r, i_v])):.1f} dB")
    print(f"  peak clutter CNR = {20 * np.log10(abs(snr_dc[:, i_0]).max()):.1f} dB")

    rdm.plot_rdm_snr(
        rdot_axis,
        r_axis,
        snr_dc,
        f"Clutter RDM, range_rate_std = {range_rate_std} m/s",
        cbar_min=0,
    )

plt.show()
