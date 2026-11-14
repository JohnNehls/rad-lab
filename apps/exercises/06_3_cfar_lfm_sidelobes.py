#!/usr/bin/env python
"""CA-CFAR on an unweighted LFM RDM: detections land on the range sidelobes.

An LFM (chirp) pulse compresses to a sinc-like range response whose peak
sidelobes sit only ~-13.2 dB below the mainlobe.  CFAR sets its threshold from
the local noise floor and knows nothing of pulse shape, so around a strong
target it fires not just on the mainlobe but on the sidelobes too -- a column
of spurious "targets" in range where there is really one.

This is the problem.  Exercise 6_4 (LFM range windowing) is the fix, and 6_5
carries the fix into the CFAR-variant comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm
from rad_lab.pulse_doppler_radar import Radar
from rad_lab.waveform import lfm_waveform
from rad_lab.returns import Target, Return
from rad_lab.cfar import cfar_2d, plot_cfar

# -- Waveform: 1 us LFM at 40 MHz (TB = T*B = 40); shared with exercise 6_4 --
bw = 40e6  # Hz
waveform = lfm_waveform(bw, T=1.0e-6, chirp_up_down=1)

# -- Radar --
radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=200e3,
    dwell_time=2e-3,
)

# -- One strong target; its ~-13 dB range sidelobes clear the noise floor --
target = Target(range=5.0e3, range_rate=300.0, rcs=30.0)

# Plain matched filter (range_window defaults to "none"), so the LFM sidelobes
# survive into the RDM.
rdot_axis, r_axis, dc = rdm.gen(radar, waveform, [Return(target=target)], plot=False)

# -- CA-CFAR --
PFA = 1e-6
dets, _ = cfar_2d(
    dc,
    n_guard_range=2,
    n_guard_doppler=2,
    n_train_range=8,
    n_train_doppler=8,
    pfa=PFA,
    method="CA",
)

# Only one target exists, so any detection well away from it is a range-sidelobe
# false alarm.
R_UNAMB = 3e8 / (2 * radar.prf)  # the 5 km target aliases into the map
r_row = np.argmin(np.abs(r_axis - target.range % R_UNAMB))
d_col = np.argmin(np.abs(rdot_axis - target.range_rate))
dr = np.abs(np.arange(dc.shape[0])[:, None] - r_row)
dd = np.abs(np.arange(dc.shape[1])[None, :] - d_col)
mainlobe_box = (dr <= 4) & (dd <= 4)
n_spurious = int((dets & ~mainlobe_box).sum())

print("## CA-CFAR on an unweighted LFM RDM ##")
print(f"  {int(dets.sum())} detections total, {n_spurious} on range sidelobes")
print("  -> exercise 6_4 suppresses these by weighting the matched filter")

plot_cfar(
    rdot_axis,
    r_axis,
    dc,
    dets,
    title=f"CA-CFAR on an unweighted LFM (Pfa={PFA:.0e}): {n_spurious} sidelobe detections",
)

plt.show()
