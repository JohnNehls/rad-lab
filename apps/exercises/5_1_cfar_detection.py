#!/usr/bin/env python
"""CFAR detection on a range-Doppler map.

Generate an RDM with three targets at different ranges and velocities, then
apply Cell-Averaging CFAR (CA-CFAR) to detect them.  The exercise shows:

1. The raw RDM with noise floor and target peaks.
2. CA-CFAR detection markers overlaid on the RDM.
3. A comparison of CA-CFAR, GOCA-CFAR, and SOCA-CFAR on the same RDM.

Key takeaways:
- CFAR adapts the detection threshold to the local noise level, maintaining a
  constant false alarm rate without requiring a fixed threshold.
- Guard cells prevent signal energy from leaking into the noise estimate.
- GOCA raises the threshold at clutter edges (fewer false alarms, slightly
  lower detection probability); SOCA lowers it (better detection in
  non-homogeneous backgrounds, but more false alarms at edges).
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm
from rad_lab.pulse_doppler_radar import Radar
from rad_lab.waveform import lfm_waveform
from rad_lab.returns import Target, Return
from rad_lab.cfar import cfar_2d, plot_cfar

# -- Waveform --
bw = 10e6  # Hz
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

# -- Three targets at different ranges and velocities --
return_list = [
    Return(target=Target(range=3.0e3, range_rate=0, rcs=10)),
    Return(target=Target(range=5.0e3, range_rate=-500, rcs=1)),
    Return(target=Target(range=4.0e3, range_rate=1e3, rcs=100)),
]

# -- Generate the RDM (suppress default plot) --
rdot_axis, r_axis, total_dc, signal_dc = rdm.gen(radar, waveform, return_list, plot=False)


# -- CA-CFAR detection --
PFA = 1e-5
print("## CA-CFAR detection ##")
detections, threshold = cfar_2d(
    total_dc,
    n_guard_range=3,
    n_guard_doppler=3,
    n_train_range=10,
    n_train_doppler=10,
    pfa=PFA,
    method="CA",
)
n_det = detections.sum()
print(f"  Pfa={PFA:e} → {n_det} cells detected")
plot_cfar(rdot_axis, r_axis, total_dc, detections, title=f"CA-CFAR Detections (Pfa={PFA:.0e})")

# -- Compare CFAR variants side by side --
print("\n## CFAR variant comparison ##")
cfar_params = dict(
    n_guard_range=3,
    n_guard_doppler=3,
    n_train_range=10,
    n_train_doppler=10,
    pfa=1e-6,
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"CFAR Variant Comparison (Pfa={cfar_params['pfa']:.0e})")

for ax, method in zip(axes, ["CA", "GOCA", "SOCA"]):
    dets, _ = cfar_2d(total_dc, method=method, **cfar_params)
    n = dets.sum()
    print(f"  {method}: {n} cells detected")

    magnitude = np.abs(total_dc)
    magnitude[magnitude == 0] = 1e-30
    plot_data = 20 * np.log10(magnitude / magnitude.max())

    mesh = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, plot_data, shading="auto")
    mesh.set_clim(-60, 0)

    det_r, det_d = np.where(dets)
    if det_r.size > 0:
        ax.plot(rdot_axis[det_d] * 1e-3, r_axis[det_r] * 1e-3, "rx", markersize=4)

    ax.set_title(f"{method}-CFAR ({n} cells detected)")
    ax.set_xlabel("Range Rate [km/s]")
    ax.set_ylabel("Range [km]")

fig.tight_layout()

plt.show()
