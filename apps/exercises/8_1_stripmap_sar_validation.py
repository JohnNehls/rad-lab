#!/usr/bin/env python
"""Validate an un-windowed stripmap SAR example

Generates three figures:
  1) sar_full.png: full SAR image from sar.gen
  2) sar_psf_zoom.png: ±150 m range, ±15 m cross-range zoom around each target
     - we are to see sinc-like patterns in rang and cross-range axes.
  3) sar_cr_cuts.png Cross-range cuts through the peak range bin of each target.
     - compares measured the -3 dB mainlobe width and to the theoretical
       resolution λR/(2L) for an un-windowed SAR image.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import sar, SarRadar, SarTarget, lfm_waveform

# -- Common SAR setup --
bw = 5e6
waveform = lfm_waveform(bw, T=10e-6, chirp_up_down=1)

sar_radar = SarRadar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=8000,
    platform_velocity=100,
    aperture_length=50,
    platform_altitude=5e3,
)

targets = [
    SarTarget(position=[-5, 0.1e3, 0], rcs=10),
    SarTarget(position=[0, 3e3, 0], rcs=10),
    SarTarget(position=[5, 5e3, 0], rcs=10),
]

cross_range, slant_range, total_dc = sar.gen(
    sar_radar, waveform, targets, seed=0, plot=False, window="none"
)

# -- 1) Full SAR image --
fig, ax = sar.plot_sar_image(cross_range, slant_range, total_dc, "Focused SAR Image")

# -- 2) Zoomed PSF around each target --
mag = np.abs(total_dc)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (tgt, ax) in enumerate(zip(targets, axes)):
    exp_sr = np.sqrt(tgt.position[0] ** 2 + tgt.position[1] ** 2 + sar_radar.platform_altitude**2)
    sr_mask = np.abs(slant_range - exp_sr) < 150
    cr_mask = np.abs(cross_range - tgt.position[0]) < 15

    sub = mag[np.ix_(sr_mask, cr_mask)]
    sub_db = 20 * np.log10(sub / mag.max() + 1e-30)
    sr_sub = slant_range[sr_mask]
    cr_sub = cross_range[cr_mask]

    mesh = ax.pcolormesh(cr_sub, sr_sub / 1e3, sub_db, vmin=-40, vmax=0)
    ax.set_title(f"Target {i}: x={tgt.position[0]}")
    ax.set_xlabel("Cross-Range [m]")
    ax.set_ylabel("Slant Range [km]")
    fig.colorbar(mesh, ax=ax)

fig.tight_layout()

# -- 3) cross-range cuts around each target --
mag = np.abs(total_dc)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (tgt, ax) in enumerate(zip(targets, axes)):
    exp_sr = np.sqrt(tgt.position[0] ** 2 + tgt.position[1] ** 2 + sar_radar.platform_altitude**2)

    # Find peak range bin
    peak_r_idx = np.argmin(np.abs(slant_range - exp_sr))
    for offset in [-1, 0, 1]:
        idx = peak_r_idx + offset
        if 0 <= idx < len(slant_range) and mag[idx, :].max() > mag[peak_r_idx, :].max():
            peak_r_idx = idx

    row = mag[peak_r_idx, :]
    row_db = 20 * np.log10(row / row.max() + 1e-30)
    cr_mask = np.abs(cross_range - tgt.position[0]) < 8

    ax.plot(cross_range[cr_mask], row_db[cr_mask])
    ax.set_xlabel("Cross-Range [m]")
    ax.set_ylabel("dB")
    ax.set_ylim(-40, 3)
    ax.grid(True)

    # Measure -3 dB width
    above_3db = cross_range[row_db > -3]
    theoretical = sar_radar.wavelength * slant_range[peak_r_idx] / (2 * sar_radar.aperture_length)
    if len(above_3db) > 0:
        width = above_3db[-1] - above_3db[0]
        ax.set_title(
            f"Target {i}: x={tgt.position[0]}m\n"
            f"-3dB width={width:.2f}m (theory={theoretical:.2f}m)"
        )
    else:
        ax.set_title(f"Target {i}: x={tgt.position[0]}m")

fig.suptitle("Cross-range cuts at peak range bin", fontsize=14)
fig.tight_layout()

plt.show()
