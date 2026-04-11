#!/usr/bin/env python
"""Multiple-PRF ambiguity resolution.

Demonstrates how running the same scene at three different PRFs lets the radar
resolve range and range-rate ambiguities that a single PRF cannot.

A target is placed beyond the unambiguous range *and* unambiguous range-rate of
each individual PRF.  Each RDM shows the target at a different aliased position.
The coincidence resolver finds the unique true range and range-rate that aliases
consistently across all three PRFs.

Key takeaways:
- A single PRF folds targets outside its unambiguous window back into the
  display — two targets at very different ranges or velocities can appear
  in the same RDM cell.
- With N>=2 PRFs whose unambiguous intervals differ, the combined unambiguous
  window is much larger (LCM of the individual windows).
- The coincidence test searches candidate true values and checks which ones
  alias consistently into the observed bin at every PRF.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm, multiprf
from rad_lab.pulse_doppler_radar import (
    Radar,
    range_unambiguous,
    range_rate_pm_unambiguous,
    range_aliased,
    range_rate_aliased_prf_f0,
)
from rad_lab.waveform import lfm_waveform
from rad_lab.returns import Target, Return

# -- Waveform --
bw = 5e6  # Hz
waveform = lfm_waveform(bw, T=1e-6, chirp_up_down=1)

# -- Common radar parameters (PRF varies per CPI) --
fcar = 10e9  # Hz
common = dict(
    fcar=fcar,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (5 / 10),
    total_losses=10 ** (3 / 10),
    dwell_time=1.6e-3,
)

# -- Three PRFs (chosen to evenly divide the sample rate) --
# R_ua:    7.5 km,  6.0 km,  3.0 km
# rdot_ua: +-150,   +-187.5, +-375 m/s
prfs = [20e3, 25e3, 50e3]

# -- Target: beyond unambiguous range and range-rate for the lower PRFs --
tgt_range = 10e3  # 10 km — aliases at all three PRFs
tgt_rdot = -200  # m/s  — aliases at PRFs 20 and 25 kHz
tgt_rcs = 1000  # 30 dBsm

return_list = [Return(target=Target(range=tgt_range, range_rate=tgt_rdot, rcs=tgt_rcs))]

# -- Print ambiguity windows --
print("=" * 60)
print("Multiple-PRF Ambiguity Resolution")
print("=" * 60)
print(f"\nTrue target: range = {tgt_range / 1e3:.1f} km, range-rate = {tgt_rdot:.1f} m/s\n")

for prf in prfs:
    r_ua = range_unambiguous(prf)
    rdot_ua = range_rate_pm_unambiguous(prf, fcar)
    r_alias = range_aliased(tgt_range, prf)
    rdot_alias = range_rate_aliased_prf_f0(tgt_rdot, prf, fcar)
    print(f"PRF = {prf / 1e3:.0f} kHz:")
    print(f"  R_ua = {r_ua / 1e3:.2f} km,  rdot_ua = ±{rdot_ua:.1f} m/s")
    print(f"  Aliased range = {r_alias / 1e3:.2f} km,  aliased rdot = {rdot_alias:.1f} m/s")

r_combined = multiprf.combined_unambiguous_range(prfs)
rdot_combined = multiprf.combined_unambiguous_range_rate(prfs, fcar)
print(f"\nCombined unambiguous range:      {r_combined / 1e3:.1f} km")
print(f"Combined unambiguous range-rate: ±{rdot_combined:.1f} m/s")

# -- Generate an RDM at each PRF --
rdot_axes = []
r_axes = []
rdms = []

for prf in prfs:
    radar = Radar(prf=prf, **common)
    rdot_axis, r_axis, total_dc, _ = rdm.gen(radar, waveform, return_list, snr=True, plot=False)
    rdot_axes.append(rdot_axis)
    r_axes.append(r_axis)
    rdms.append(total_dc)

# -- Plot the three RDMs side by side --
multiprf.plot_multiprf_rdms(rdot_axes, r_axes, rdms, prfs)

# -- Find the peak (aliased detection) in each RDM --
observed_ranges = []
observed_rdots = []

print("\n--- Detections ---")
for i, prf in enumerate(prfs):
    mag = np.abs(rdms[i])
    peak_idx = np.unravel_index(mag.argmax(), mag.shape)
    r_det = r_axes[i][peak_idx[0]]
    rdot_det = rdot_axes[i][peak_idx[1]]
    observed_ranges.append(r_det)
    observed_rdots.append(rdot_det)
    print(f"  PRF {prf / 1e3:.0f} kHz: range = {r_det / 1e3:.2f} km, rdot = {rdot_det:.1f} m/s")

# -- Resolve ambiguities --
range_res = 3e8 / (2 * bw)
rdot_res = max(abs(ax[1] - ax[0]) for ax in rdot_axes)

resolved_r = multiprf.resolve_range(observed_ranges, prfs, tolerance=2 * range_res)
resolved_rdot = multiprf.resolve_range_rate(observed_rdots, prfs, fcar, tolerance=2 * rdot_res)

print("\n--- Resolved ---")
if len(resolved_r) > 0:
    best_r = resolved_r[np.argmin(np.abs(resolved_r - tgt_range))]
    print(f"  Range:      {best_r / 1e3:.2f} km  (true: {tgt_range / 1e3:.1f} km)")
    range_err = abs(best_r - tgt_range)
    print(f"  Range error: {range_err:.1f} m  {'PASS' if range_err < 2 * range_res else 'FAIL'}")
    print(f"  Candidates: {len(resolved_r)}")
else:
    print("  Range: no consistent solution found")

if len(resolved_rdot) > 0:
    best_rdot = resolved_rdot[np.argmin(np.abs(resolved_rdot - tgt_rdot))]
    print(f"  Range-rate: {best_rdot:.1f} m/s  (true: {tgt_rdot:.1f} m/s)")
    rdot_err = abs(best_rdot - tgt_rdot)
    print(f"  Rdot error:  {rdot_err:.1f} m/s  {'PASS' if rdot_err < 2 * rdot_res else 'FAIL'}")
    print(f"  Candidates: {len(resolved_rdot)}")
else:
    print("  Range-rate: no consistent solution found")

# -- Plot the resolution diagram --
multiprf.plot_resolved(observed_ranges, observed_rdots, resolved_r, resolved_rdot, prfs, fcar)

plt.show()
