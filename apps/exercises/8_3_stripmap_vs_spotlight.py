#!/usr/bin/env python
"""Side-by-side comparison of stripmap and spotlight SAR.

Runs both modes on the same scene and plots:
  - Top row: focused SAR images (stripmap vs spotlight)
  - Bottom row: cross-range cuts through the centre target with -3 dB
    width annotations

The spotlight aperture is 4× longer, yielding ~4× finer cross-range
resolution.
"""

import matplotlib.pyplot as plt
import numpy as np
from rad_lab import sar, SarRadar, SarTarget, lfm_waveform

# -- Shared waveform and targets --
bw = 5e6
waveform = lfm_waveform(bw, T=10e-6, chirp_up_down=1)

targets = [
    SarTarget(position=[-5, 0.1e3, 0], rcs=10),
    SarTarget(position=[0, 3e3, 0], rcs=10),
    SarTarget(position=[5, 5e3, 0], rcs=10),
]

# -- Shared RF parameters --
common = dict(
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
    platform_altitude=5e3,
)

# -- Stripmap configuration --
strip_radar = SarRadar(aperture_length=50, **common)

# -- Spotlight configuration (4× longer aperture, beam steered to centre target) --
spot_radar = SarRadar(
    aperture_length=200,
    scene_center=[0, 3e3, 0],
    beamwidth=0.06,
    **common,
)

# -- Run both modes --
print("Running stripmap SAR ...")
cr_s, sr_s, sig_s = sar.gen(strip_radar, waveform, targets, seed=0, plot=False)

print("Running spotlight SAR ...")
cr_p, sr_p, sig_p = sar.gen(spot_radar, waveform, targets, seed=0, plot=False)

# -- Analyse centre target (index 1) --
centre_tgt = targets[1]
exp_sr = np.sqrt(
    centre_tgt.position[0] ** 2 + centre_tgt.position[1] ** 2 + common["platform_altitude"] ** 2
)


def measure_peak(cr, sr, sig, exp_sr):
    """Return cross-range cut in dB and -3 dB width at the peak range bin."""
    mag = np.abs(sig)
    peak_r = np.argmin(np.abs(sr - exp_sr))
    # Refine to strongest neighbouring bin
    for offset in [-1, 1]:
        idx = peak_r + offset
        if 0 <= idx < len(sr) and mag[idx, :].max() > mag[peak_r, :].max():
            peak_r = idx
    row = mag[peak_r, :]
    row_db = 20 * np.log10(row / row.max() + 1e-30)
    above = cr[row_db > -3]
    width = above[-1] - above[0] if len(above) > 0 else float("nan")
    return row_db, width, sr[peak_r]


cut_s, w_s, sr_peak_s = measure_peak(cr_s, sr_s, sig_s, exp_sr)
cut_p, w_p, sr_peak_p = measure_peak(cr_p, sr_p, sig_p, exp_sr)

theory_s = strip_radar.wavelength * sr_peak_s / (2 * strip_radar.aperture_length)
theory_p = spot_radar.wavelength * sr_peak_p / (2 * spot_radar.aperture_length)

print("\nCentre target cross-range -3 dB width:")
print(f"  Stripmap:  {w_s:.2f} m  (theory {theory_s:.2f} m)")
print(f"  Spotlight: {w_p:.2f} m  (theory {theory_p:.2f} m)")
print(f"  Ratio:     {w_s / w_p:.1f}×")

# -- Plot --
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top row: SAR images (zoomed to ±15 m cross-range, target range ± 500 m)
for ax, cr, sr, sig, title in [
    (axes[0, 0], cr_s, sr_s, sig_s, "Stripmap"),
    (axes[0, 1], cr_p, sr_p, sig_p, "Spotlight"),
]:
    mag = np.abs(sig)
    mag_db = 20 * np.log10(mag / mag.max() + 1e-30)
    sr_mask = np.abs(sr - exp_sr) < 500
    cr_mask = np.abs(cr - centre_tgt.position[0]) < 15
    sub = mag_db[np.ix_(sr_mask, cr_mask)]
    mesh = ax.pcolormesh(cr[cr_mask], sr[sr_mask] / 1e3, sub, vmin=-40, vmax=0, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Cross-Range [m]")
    ax.set_ylabel("Slant Range [km]")
    fig.colorbar(mesh, ax=ax, label="dB")

# Bottom row: cross-range cuts
cr_zoom = 10  # ±10 m around target
for ax, cr, cut, w, theory, title in [
    (axes[1, 0], cr_s, cut_s, w_s, theory_s, "Stripmap"),
    (axes[1, 1], cr_p, cut_p, w_p, theory_p, "Spotlight"),
]:
    mask = np.abs(cr - centre_tgt.position[0]) < cr_zoom
    ax.plot(cr[mask], cut[mask], linewidth=1.5)
    ax.axhline(-3, color="r", linestyle="--", linewidth=0.8, label="-3 dB")
    ax.set_title(f"{title}: -3 dB width = {w:.2f} m (theory {theory:.2f} m)")
    ax.set_xlabel("Cross-Range [m]")
    ax.set_ylabel("Normalised Magnitude [dB]")
    ax.set_ylim(-40, 3)
    ax.legend()
    ax.grid(True)

fig.suptitle("Stripmap vs Spotlight SAR — Centre Target", fontsize=14)
fig.tight_layout()
# fig.savefig("stripmap_vs_spotlight.png", dpi=200)
# print("\nSaved stripmap_vs_spotlight.png")

plt.show()
