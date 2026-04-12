#!/usr/bin/env python
"""Swerling fluctuation model comparison.

Compares all five Swerling target-fluctuation models under non-coherent
integration (NCI) to illustrate the effect of RCS decorrelation rate and
degrees of freedom on detection performance.

=======  ====  ============================  ================================
Model    DOF   Decorrelation                 Physical interpretation
=======  ====  ============================  ================================
0 / V    --    Non-fluctuating (constant)    Ideal / calibration target
I        2     Scan-to-scan (slow)           Single dominant scatterer
II       2     Pulse-to-pulse (fast)         Single scatterer, fast fading
III      4     Scan-to-scan (slow)           Few comparable scatterers
IV       4     Pulse-to-pulse (fast)         Few scatterers, fast fading
=======  ====  ============================  ================================

Key takeaways:
- For a single pulse (N=1), Swerling I ≡ II and III ≡ IV — the
  decorrelation rate is irrelevant without integration.
- With NCI (N > 1), pulse-to-pulse models (II, IV) outperform their
  scan-to-scan counterparts (I, III) because each pulse provides an
  independent RCS draw, giving a diversity gain that reduces the
  effective SNR variance.
- Higher DOF (III/IV vs I/II) means less extreme RCS swings and
  therefore less SNR required at any given N.
- As N → ∞, all fluctuating models converge toward Swerling 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import detection

pfa = 1e-6

# ── Figure 1: Pd vs per-pulse SNR, N = 10, all models ────────────────────
snr_db = np.linspace(-5, 25, 100)
snr_lin = 10 ** (snr_db / 10)
n_pulses = 10

fig1, ax1 = plt.subplots(figsize=(8, 5))
curves = [
    ("Swerling 0", detection.pd_swerling0_nci(snr_lin, pfa, n_pulses)),
    ("Swerling II", detection.pd_swerling2(snr_lin, pfa, n_pulses)),
    ("Swerling IV", detection.pd_swerling4(snr_lin, pfa, n_pulses)),
    ("Swerling III", detection.pd_swerling3_nci(snr_lin, pfa, n_pulses)),
    ("Swerling I", detection.pd_swerling1_nci(snr_lin, pfa, n_pulses)),
]
for label, pd_vals in curves:
    ax1.plot(snr_db, pd_vals, label=label)

ax1.set_xlabel("Per-pulse SNR [dB]")
ax1.set_ylabel("$P_d$")
ax1.set_title(f"All Swerling Models — NCI $N = {n_pulses}$, $P_{{fa}} = {pfa:.0e}$")
ax1.set_ylim(0, 1.05)
ax1.legend()
ax1.grid(True)
fig1.tight_layout()

# ── Figure 2: Scan-to-scan vs pulse-to-pulse at varying N ────────────────
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
n_values = [1, 5, 10, 30]
snr_db2 = np.linspace(-5, 30, 100)
snr_lin2 = 10 ** (snr_db2 / 10)

for n in n_values:
    pd_sw1 = detection.pd_swerling1_nci(snr_lin2, pfa, n)
    pd_sw2 = detection.pd_swerling2(snr_lin2, pfa, n)
    ax2a.plot(snr_db2, pd_sw1, "--", label=f"Sw I,  N={n}")
    ax2a.plot(snr_db2, pd_sw2, "-", label=f"Sw II, N={n}")

ax2a.set_xlabel("Per-pulse SNR [dB]")
ax2a.set_ylabel("$P_d$")
ax2a.set_title("2 DOF: Swerling I (dashed) vs II (solid)")
ax2a.set_ylim(0, 1.05)
ax2a.legend(fontsize=8, ncol=2)
ax2a.grid(True)

for n in n_values:
    pd_sw3 = detection.pd_swerling3_nci(snr_lin2, pfa, n)
    pd_sw4 = detection.pd_swerling4(snr_lin2, pfa, n)
    ax2b.plot(snr_db2, pd_sw3, "--", label=f"Sw III, N={n}")
    ax2b.plot(snr_db2, pd_sw4, "-", label=f"Sw IV,  N={n}")

ax2b.set_xlabel("Per-pulse SNR [dB]")
ax2b.set_ylabel("$P_d$")
ax2b.set_title("4 DOF: Swerling III (dashed) vs IV (solid)")
ax2b.set_ylim(0, 1.05)
ax2b.legend(fontsize=8, ncol=2)
ax2b.grid(True)
fig2.suptitle(f"Diversity Gain from Pulse-to-Pulse Fluctuation ($P_{{fa}} = {pfa:.0e}$)")
fig2.tight_layout()

# ── Figure 3: Required per-pulse SNR vs N ─────────────────────────────────
pd_target = 0.9
n_range = np.arange(1, 31)

fig3, ax3 = plt.subplots(figsize=(8, 5))
for model, label, ls in [
    ("swerling0", "Swerling 0", "-"),
    ("swerling2", "Swerling II", "-"),
    ("swerling4", "Swerling IV", "-"),
    ("swerling3", "Swerling III", "--"),
    ("swerling1", "Swerling I", "--"),
]:
    snr_req = [detection.required_snr_nci(pd_target, pfa, int(n), model=model) for n in n_range]
    ax3.plot(n_range, snr_req, ls, label=label)

ax3.set_xlabel("Number of non-coherently integrated pulses $N$")
ax3.set_ylabel("Required per-pulse SNR [dB]")
ax3.set_title(f"Required SNR vs $N$ ($P_d = {pd_target}$, $P_{{fa}} = {pfa:.0e}$)")
ax3.legend()
ax3.grid(True)
fig3.tight_layout()

# ── Summary table ─────────────────────────────────────────────────────────
print("=" * 60)
print("Swerling Model Comparison")
print("=" * 60)
print(f"Pfa = {pfa:.0e}, Pd = {pd_target}, N = {n_pulses}")
print()
print(f"{'Model':<14} {'Req. SNR/pulse [dB]':>20}")
print("-" * 36)
for model, label in [
    ("swerling0", "Swerling 0"),
    ("swerling1", "Swerling I"),
    ("swerling2", "Swerling II"),
    ("swerling3", "Swerling III"),
    ("swerling4", "Swerling IV"),
]:
    snr = detection.required_snr_nci(pd_target, pfa, n_pulses, model=model)
    print(f"{label:<14} {snr:>20.1f}")

plt.show()
