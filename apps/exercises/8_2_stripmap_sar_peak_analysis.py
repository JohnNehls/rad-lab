#!/usr/bin/env python
"""Numerical peak analysis: prints peak locations and azimuth peak drift vs range bin.

For each target, finds the peak position and then shows how the azimuth peak
shifts across neighboring range bins.  A tilt-free PSF will have constant
azimuth peak position across range bins.
"""

import numpy as np
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

cross_range, slant_range, total_dc = sar.gen(sar_radar, waveform, targets, seed=0, plot=False)

mag = np.abs(total_dc)
mag_db = 20 * np.log10(mag / mag.max() + 1e-30)

print("=== Peak analysis ===")
print(
    f"Cross-range axis: {cross_range.min():.1f} to {cross_range.max():.1f} m, {len(cross_range)} pts"
)
print(
    f"Slant-range axis: {slant_range.min():.1f} to {slant_range.max():.1f} m, {len(slant_range)} pts"
)
print(f"Range resolution: {slant_range[1] - slant_range[0]:.2f} m")
print(f"Cross-range spacing: {cross_range[1] - cross_range[0]:.4f} m")
print()

for tgt_idx, tgt in enumerate(targets):
    exp_sr = np.sqrt(tgt.position[0] ** 2 + tgt.position[1] ** 2 + sar_radar.platform_altitude**2)
    print(f"Target {tgt_idx}: pos={tgt.position}, expected slant range={exp_sr:.1f} m")

    # Find peak in neighborhood
    sr_mask = np.abs(slant_range - exp_sr) < 200
    cr_mask = np.abs(cross_range - tgt.position[0]) < 15

    sub = mag[np.ix_(sr_mask, cr_mask)]
    sr_sub = slant_range[sr_mask]
    cr_sub = cross_range[cr_mask]

    peak_idx = np.unravel_index(np.argmax(sub), sub.shape)
    print(
        f"  Peak at slant_range={sr_sub[peak_idx[0]]:.1f} m, cross_range={cr_sub[peak_idx[1]]:.3f} m"
    )
    print(f"  Peak magnitude (dB): {20 * np.log10(sub.max() / mag.max()):.1f}")

    # Azimuth peak position vs range bin offset
    peak_r_idx = np.where(sr_mask)[0][peak_idx[0]]
    print("  Azimuth peak position vs range bin offset:")
    for dr in range(-3, 4):
        r_idx = peak_r_idx + dr
        if 0 <= r_idx < len(slant_range):
            row = mag[r_idx, :]
            az_peak = np.argmax(row)
            peak_db = 20 * np.log10(row[az_peak] / mag.max() + 1e-30)
            print(
                f"    range bin {dr:+d} (R={slant_range[r_idx]:.1f}m): "
                f"az peak at cr={cross_range[az_peak]:.3f}m, {peak_db:.1f} dB"
            )
    print()
