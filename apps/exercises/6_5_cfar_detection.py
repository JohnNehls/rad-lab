#!/usr/bin/env python
"""CFAR detection on a range-Doppler map (with LFM range weighting).

Generate an RDM with three targets at different ranges and velocities, then
apply Cell-Averaging CFAR (CA-CFAR) to detect them.  The matched filter is
Taylor-weighted for range-sidelobe control (see exercise 6_4), so the LFM's
~-13.2 dB range sidelobes stay below the CFAR threshold and do not clutter the
map with spurious detections -- isolating the behaviour of the CFAR variants
themselves.  The exercise produces four figures:

1. The raw RDM with noise floor and target peaks.
2. CA-CFAR detection markers overlaid on the RDM.
3. A comparison of CA-CFAR, GOCA-CFAR, and SOCA-CFAR on the same RDM.
4. A clutter-edge scene where the variants actually differ: GOCA suppresses
   the false alarms CA and SOCA fire along the edge, while SOCA detects a
   weak target near the edge that clutter in the training window masks from
   CA and GOCA.

Key takeaways:
- CFAR adapts the detection threshold to the local noise level, maintaining a
  constant false alarm rate without requiring a fixed threshold.
- Guard cells prevent signal energy from leaking into the noise estimate.
- In homogeneous noise the variants only shift the threshold (SOCA lowest,
  GOCA highest); their real differences appear at a clutter edge.  Inside the
  clutter, a window straddling the edge drags the CA threshold down (false
  alarms; SOCA far worse), while GOCA keys on the clutter half and stays
  clean.  Just outside the clutter, the same straddling inflates the CA and
  GOCA thresholds (masking weak targets), while SOCA keys on the clear half
  and still detects.  CA is the compromise; no variant wins everywhere.
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

# Taylor weighting on the matched-filter replica, to suppress the LFM range
# sidelobes (see exercise 6_4).  Applied to every RDM generated below.
RANGE_WINDOW = "taylor"
RANGE_WINDOW_KWARGS = {"nbar": 5, "sll": 35}

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
rdot_axis, r_axis, total_dc = rdm.gen(
    radar,
    waveform,
    return_list,
    plot=False,
    range_window=RANGE_WINDOW,
    range_window_kwargs=RANGE_WINDOW_KWARGS,
)


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


def plot_rdm_with_detections(ax, rdot_axis, r_axis, dc, dets, title):
    """Panel plot: RDM in dB with detection markers overlaid.  Returns the mesh
    so the caller can attach a shared colorbar."""
    magnitude = np.abs(dc)
    magnitude[magnitude == 0] = np.finfo(float).tiny
    plot_data = 20 * np.log10(magnitude / magnitude.max())

    mesh = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, plot_data, shading="auto")
    mesh.set_clim(-60, 0)

    det_r, det_d = np.where(dets)
    if det_r.size > 0:
        ax.plot(
            rdot_axis[det_d] * 1e-3, r_axis[det_r] * 1e-3, "rx", markersize=4, label="detections"
        )

    ax.set_title(title)
    ax.set_xlabel("Range Rate [km/s]")
    ax.set_ylabel("Range [km]")
    return mesh


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"CFAR Variant Comparison (Pfa={cfar_params['pfa']:.0e})")

for ax, method in zip(axes, ["CA", "GOCA", "SOCA"]):
    dets, _ = cfar_2d(total_dc, method=method, **cfar_params)
    n = dets.sum()
    print(f"  {method}: {n} cells detected")
    mesh = plot_rdm_with_detections(
        ax, rdot_axis, r_axis, total_dc, dets, f"{method}-CFAR ({n} cells detected)"
    )

axes[-1].legend(loc="upper right")
fig.colorbar(mesh, ax=axes, label="Normalised Magnitude [dB]")

# -- Scene 2: clutter edge, where the variants actually differ --
# The GOCA/SOCA halves split leading/lagging in *range* (see cfar_2d), so the
# clutter edge is oriented in range: everything beyond CLUTTER_START is
# clutter, at all Dopplers.  Two targets probe the two failure modes:
#   - a strong control target in the clear, well away from the edge
#   - a weak target just outside the edge, close enough that the CFAR
#     training window straddles the boundary
# Note that cfar_2d pads by wrapping, so the top of the map borders the clear
# near-range rows — a second clutter edge with the same false-alarm behavior.
print("\n## Scene 2: clutter edge ##")

CLUTTER_START = 400.0  # m
CNR_DB = 25  # clutter-to-noise ratio [dB]

# A looser Pfa than scene 1 so CA's false-alarm inflation at the edge is
# visible on a map this size (at 1e-6 it amounts to ~1 false alarm)
cfar_params2 = {**cfar_params, "pfa": 1e-4}

control_tgt = Target(range=250.0, range_rate=-800.0, rcs=1e-3)
edge_tgt = Target(range=CLUTTER_START - 30.0, range_rate=400.0, rcs=1e-6)

rdot_axis2, r_axis2, dc2 = rdm.gen(
    radar,
    waveform,
    [Return(target=control_tgt), Return(target=edge_tgt)],
    plot=False,
    range_window=RANGE_WINDOW,
    range_window_kwargs=RANGE_WINDOW_KWARGS,
)

# Inject clutter: complex Gaussian noise CNR_DB above the thermal floor for
# all ranges beyond CLUTTER_START (the median is robust to the target peaks)
rng = np.random.default_rng(1)
noise_power = np.median(np.abs(dc2) ** 2)
clutter_rows = r_axis2 > CLUTTER_START
sigma = np.sqrt(noise_power * 10 ** (CNR_DB / 10) / 2)
shape = (int(clutter_rows.sum()), dc2.shape[1])
dc2[clutter_rows, :] += sigma * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

# Expected cell of the weak target next to the edge, and its measured SNR
edge_r = np.argmin(np.abs(r_axis2 - edge_tgt.range))
edge_d = np.argmin(np.abs(rdot_axis2 - edge_tgt.range_rate))
edge_power = np.abs(dc2[edge_r - 2 : edge_r + 3, edge_d - 2 : edge_d + 3]) ** 2
edge_snr_db = 10 * np.log10(edge_power.max() / noise_power)
print(f"  edge target SNR: {edge_snr_db:.1f} dB")

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle(f"CFAR at a Clutter Edge (Pfa={cfar_params2['pfa']:.0e}, CNR={CNR_DB} dB)")

for ax, method in zip(axes2, ["CA", "GOCA", "SOCA"]):
    dets, _ = cfar_2d(dc2, method=method, **cfar_params2)

    # No targets lie inside the clutter band, so every detection there is a
    # false alarm.  This counts only the in-clutter false alarms; edge-induced
    # false alarms that fall in the clear rows just below the edge are not
    # included, so it is a lower bound on the total edge effect.
    n_fa = dets[clutter_rows, :].sum()
    edge_hit = dets[edge_r - 2 : edge_r + 3, edge_d - 2 : edge_d + 3].any()

    print(f"  {method}: {n_fa} false alarms inside clutter, edge target detected: {edge_hit}")
    fa_label = f"{n_fa} in-clutter FA{'' if n_fa == 1 else 's'}"
    mesh = plot_rdm_with_detections(
        ax,
        rdot_axis2,
        r_axis2,
        dc2,
        dets,
        f"{method}-CFAR ({fa_label}, edge target {'detected' if edge_hit else 'MASKED'})",
    )
    ax.axhline(CLUTTER_START * 1e-3, color="w", linestyle="--", linewidth=1, label="clutter edge")
    ax.plot(
        edge_tgt.range_rate * 1e-3,
        edge_tgt.range * 1e-3,
        "o",
        color="w",
        fillstyle="none",
        markersize=10,
        label="weak target",
    )

axes2[-1].legend(loc="upper right")
fig2.colorbar(mesh, ax=axes2, label="Normalised Magnitude [dB]")

plt.show()
