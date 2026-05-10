#!/usr/bin/env python
"""Demonstrates Range Cell Migration Correction (RCMC).

Runs ``sar.gen`` twice on a long-aperture / close-range collection whose
peak range migration spans multiple range cells, then prints the
azimuth peak position vs range bin offset for each.  Without RCMC the
azimuth peak tilts across neighbouring range bins (the RCM signature);
with RCMC it sits at a constant cross-range position.

The first ``rcmc=True`` run is also called with ``debug=True`` so the
range-Doppler map is plotted before and after the correction.  In the
"before" panel each target traces a curved hyperbola across Doppler
frequency; in the "after" panel those curves collapse to straight
horizontal lines at each target's closest-approach range.
"""

import matplotlib.pyplot as plt
import numpy as np
from rad_lab import sar, SarRadar, SarTarget, lfm_waveform
from rad_lab import constants as c

# -- SAR setup tuned so peak migration spans ~2 range cells --
# migration_m  = aperture^2 / (8 * R0)
# range_cell_m = c / (2 * sample_rate)  with sample_rate = 2 * bw
# => migration_cells = aperture^2 * bw / (2 * c * R0)
#
# Datacube memory ~ 16 * aperture * bw / v (PRF cancels).  PRF must
# stay above the target's Doppler bandwidth 2*v*aperture/(R0*lambda)
# to avoid azimuth aliasing.  Aperture/R0 is kept ~0.33 to stay inside
# the small-squint regime where the RDA's parabolic migration model
# holds.  This config gives ~2 cells of migration in a ~140 MB
# datacube (~7x smaller than a 5-cell demo would need).
bw = 10e6
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
    prf=16000,
    platform_velocity=600,
    aperture_length=600,
    platform_altitude=1500,
)

targets = [
    SarTarget(position=[0, 1500, 0], rcs=10),
]

R0 = float(np.sqrt(targets[0].position[1] ** 2 + sar_radar.platform_altitude**2))
range_cell_m = c.C / (2 * sar_radar.sample_rate)
peak_migration_m = sar_radar.aperture_length**2 / (8 * R0)
print(
    f"Predicted peak migration: {peak_migration_m:.2f} m "
    f"({peak_migration_m / range_cell_m:.2f} range cells)\n"
)


def run_peak_analysis(
    label: str, rcmc: bool, debug: bool = False, plot: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("=" * 70)
    print(f"  {label}  (rcmc={rcmc})")
    print("=" * 70)
    cross_range, slant_range, total_dc = sar.gen(
        sar_radar, waveform, targets, seed=0, plot=plot, debug=debug, rcmc=rcmc
    )

    mag = np.abs(total_dc)

    print(
        f"Cross-range axis: {cross_range.min():.1f} to {cross_range.max():.1f} m, "
        f"{len(cross_range)} pts"
    )
    print(
        f"Slant-range axis: {slant_range.min():.1f} to {slant_range.max():.1f} m, "
        f"{len(slant_range)} pts"
    )
    print(f"Range resolution: {slant_range[1] - slant_range[0]:.2f} m")
    print(f"Cross-range spacing: {cross_range[1] - cross_range[0]:.4f} m")
    print()

    for tgt_idx, tgt in enumerate(targets):
        exp_sr = float(
            np.sqrt(tgt.position[0] ** 2 + tgt.position[1] ** 2 + sar_radar.platform_altitude**2)
        )
        print(f"Target {tgt_idx}: pos={tgt.position}, expected slant range={exp_sr:.1f} m")

        sr_mask = np.abs(slant_range - exp_sr) < 200
        cr_mask = np.abs(cross_range - tgt.position[0]) < 30

        sub = mag[np.ix_(sr_mask, cr_mask)]
        sr_sub = slant_range[sr_mask]
        cr_sub = cross_range[cr_mask]

        peak_idx = np.unravel_index(np.argmax(sub), sub.shape)
        print(
            f"  Peak at slant_range={sr_sub[peak_idx[0]]:.1f} m, "
            f"cross_range={cr_sub[peak_idx[1]]:.3f} m"
        )
        print(f"  Peak magnitude (dB): {20 * np.log10(sub.max() / mag.max()):.1f}")

        peak_r_idx = np.where(sr_mask)[0][peak_idx[0]]
        print("  Azimuth peak position vs range bin offset:")
        for dr in range(-3, 4):
            r_idx = peak_r_idx + dr
            if 0 <= r_idx < len(slant_range):
                row = mag[r_idx, :]
                az_peak = int(np.argmax(row))
                peak_db = 20 * np.log10(row[az_peak] / mag.max() + 1e-30)
                print(
                    f"    range bin {dr:+d} (R={slant_range[r_idx]:.1f}m): "
                    f"az peak at cr={cross_range[az_peak]:.3f}m, {peak_db:.1f} dB"
                )
        print()
    return cross_range, slant_range, total_dc


# First pass: numerical only, no figures (keeps peak RAM down — the
# matplotlib pcolormeshes from this run would otherwise still be alive
# while the second run allocates its own datacube + 6 debug figures).
run_peak_analysis("WITHOUT RCMC", rcmc=False, plot=False, debug=False)
# Second pass: full debug + plots so the user sees the RDM before/after.
run_peak_analysis("WITH RCMC + debug RDM panels", rcmc=True, plot=True, debug=True)

print("Rendering figures (close all windows to exit)...")
plt.show()
