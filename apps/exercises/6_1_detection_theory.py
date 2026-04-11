#!/usr/bin/env python
"""Detection theory fundamentals.

Demonstrates the relationship between SNR, probability of false alarm (Pfa),
and probability of detection (Pd) for non-fluctuating and fluctuating targets.

Produces four figures:
1. Pd vs SNR for Swerling 0, I, and III at a fixed Pfa.
2. ROC curves (Pd vs Pfa) for Swerling 0 at several SNR values.
3. Required SNR vs number of non-coherently integrated pulses,
   comparing exact numerical results with Albersheim's approximation.
4. Pd vs SNR comparing coherent single-look with NCI for several N.

Key takeaways:
- A non-fluctuating target (Swerling 0) has the steepest Pd-vs-SNR curve.
  Fluctuating targets (Swerling I, III) require higher average SNR for the
  same Pd because the instantaneous RCS can be much lower than the average.
- Swerling III (4 DOF) performs better than Swerling I (2 DOF) because
  its RCS is less likely to fade to zero.
- Non-coherent integration reduces the per-pulse SNR requirement at the
  cost of needing multiple independent looks.
- Albersheim's closed-form approximation is useful for quick estimates but
  can deviate significantly from the exact result, especially for small N.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab.detection import (
    pd_swerling0,
    pd_swerling0_nci,
    required_snr,
    plot_pd_vs_snr,
    plot_roc,
    plot_required_snr_vs_n,
)

# -- Parameters --
pfa = 1e-6
snr_db = np.arange(0, 25, 0.2)

# ----------------------------------------------------------------
# Figure 1: Pd vs SNR for Swerling 0, I, III
# ----------------------------------------------------------------
plot_pd_vs_snr(snr_db, pfa)

# ----------------------------------------------------------------
# Figure 2: ROC curves for Swerling 0 at several SNR values
# ----------------------------------------------------------------
plot_roc([5, 8, 10, 13, 15], model="swerling0")

# ----------------------------------------------------------------
# Figure 3: Required SNR_1 vs N (exact + Albersheim)
# ----------------------------------------------------------------
plot_required_snr_vs_n(pd=0.9, pfa=pfa, n_pulses_range=np.arange(1, 65))

# ----------------------------------------------------------------
# Figure 4: Coherent single-look vs NCI
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
snr_lin = 10 ** (snr_db / 10)

ax.plot(snr_db, pd_swerling0(snr_lin, pfa), label="Coherent (single look)")
for n in [5, 10, 20]:
    pd_nci = pd_swerling0_nci(snr_lin, pfa, n)
    ax.plot(snr_db, pd_nci, "--", label=f"NCI, N={n}")

ax.set_xlabel("Per-pulse SNR [dB]")
ax.set_ylabel("$P_d$")
ax.set_title(f"Coherent vs Non-Coherent Integration ($P_{{fa}} = {pfa:.0e}$)")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True)
fig.tight_layout()

# ----------------------------------------------------------------
# Print required SNR table
# ----------------------------------------------------------------
print("=" * 50)
print("Detection Theory Summary")
print("=" * 50)
print(f"\nPfa = {pfa:.0e}\n")

print("Required post-integration SNR for Pd = 0.9:")
for model in ["swerling0", "swerling1", "swerling3"]:
    snr = required_snr(0.9, pfa, model)
    print(f"  {model:12s}: {snr:.2f} dB")

print("\nRequired per-pulse SNR for Pd = 0.9 (Swerling 0, NCI):")
for n in [1, 5, 10, 20, 50]:
    from rad_lab.detection import required_snr_nci

    snr = required_snr_nci(0.9, pfa, n)
    print(f"  N = {n:3d}: {snr:.2f} dB")

plt.show()
