#!/usr/bin/env python
"""Monte Carlo validation of Swerling NCI detection theory.

Generates noisy baseband pulses with Swerling-fluctuated amplitude,
square-law detects each pulse, sums N of them (non-coherent integration),
thresholds, and counts detections over many trials.  The empirical Pd is
compared to the closed-form theoretical curves from ``rad_lab.detection``.

Signal model per pulse
----------------------
.. math::

    s_i     &= \sqrt{\text{SNR} \cdot \sigma_i / \bar\sigma}\;
               e^{j\theta_i}  \\
    x_i     &= s_i + n_i, \qquad n_i \sim \mathcal{CN}(0,1)   \\
    y_i     &= |x_i|^2         \qquad\text{(square-law detection)}

NCI test statistic: :math:`T = \sum_{i=1}^{N} y_i`

Threshold: :math:`V_T = \chi^2_{2N}^{-1}(P_{fa}) / 2` via
``detection.threshold_factor_nci``.

Detection: :math:`T > V_T`.

RCS draws per Swerling model
-----------------------------
=======  ====  ==========================================
Model    DOF   Draw rule
=======  ====  ==========================================
0        --    :math:`\sigma_i = \bar\sigma` (constant)
I        2     One :math:`\sigma \sim \text{Exp}(\bar\sigma)`, same all pulses
II       2     Independent :math:`\sigma_i \sim \text{Exp}(\bar\sigma)` per pulse
III      4     One :math:`\sigma \sim \Gamma(2,\,\bar\sigma/2)`, same all pulses
IV       4     Independent :math:`\sigma_i \sim \Gamma(2,\,\bar\sigma/2)` per pulse
=======  ====  ==========================================

Key takeaways:
- Empirical Pd matches theory to within statistical noise for all five models,
  confirming the detection module's closed-form and quadrature implementations.
- NCI is square-law-then-sum — fundamentally different from coherent Doppler FFT.
- Scan-to-scan models (I, III) draw one RCS per trial; all N pulses share it.
- Pulse-to-pulse models (II, IV) draw independent RCS per pulse.
- The false-alarm histogram confirms the chi-squared noise model and threshold.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from rad_lab import detection
from rad_lab.noise import unity_variance_complex_noise

# -- Parameters ---------------------------------------------------------------
pfa = 1e-6
n_trials = 50_000
np.random.seed(42)


# -- Monte Carlo engine -------------------------------------------------------
def monte_carlo_pd(snr_db_values, n_pulses, swerling, pfa, n_trials):
    """Run NCI Monte Carlo for one Swerling model across an SNR sweep.

    Args:
        snr_db_values: 1-D array of per-pulse SNR values [dB].
        n_pulses: Number of non-coherently integrated pulses.
        swerling: Swerling model number (0–4).
        pfa: Probability of false alarm.
        n_trials: Number of independent trials per SNR point.

    Returns:
        1-D array of empirical Pd, same length as *snr_db_values*.
    """
    vt = detection.threshold_factor_nci(pfa, n_pulses)
    pd_emp = np.empty(len(snr_db_values))

    for idx, snr_db in enumerate(snr_db_values):
        snr_lin = 10 ** (snr_db / 10)

        # -- Draw per-pulse RCS for all trials: shape (n_trials, n_pulses) --
        if swerling == 0:
            rcs_ratio = np.ones((n_trials, n_pulses))
        elif swerling == 1:
            # Scan-to-scan, 2 DOF: one Exp(1) draw per trial
            draws = np.random.exponential(1.0, size=(n_trials, 1))
            rcs_ratio = np.broadcast_to(draws, (n_trials, n_pulses))
        elif swerling == 2:
            # Pulse-to-pulse, 2 DOF: independent Exp(1) per pulse
            rcs_ratio = np.random.exponential(1.0, size=(n_trials, n_pulses))
        elif swerling == 3:
            # Scan-to-scan, 4 DOF: one Gamma(2,0.5) draw → mean 1
            draws = np.random.gamma(2, 0.5, size=(n_trials, 1))
            rcs_ratio = np.broadcast_to(draws, (n_trials, n_pulses))
        elif swerling == 4:
            # Pulse-to-pulse, 4 DOF: independent Gamma(2,0.5) per pulse
            rcs_ratio = np.random.gamma(2, 0.5, size=(n_trials, n_pulses))
        else:
            raise ValueError(f"Unknown Swerling model {swerling}")

        # -- Signal: sqrt(snr * rcs_ratio) * exp(j*theta) --
        amplitude = np.sqrt(snr_lin * rcs_ratio)
        phase = np.random.uniform(0, 2 * np.pi, size=(n_trials, n_pulses))
        signal = amplitude * np.exp(1j * phase)

        # -- Noise: CN(0, 1) --
        noise = unity_variance_complex_noise((n_trials, n_pulses))

        # -- Square-law detection + NCI --
        y = np.abs(signal + noise) ** 2  # (n_trials, n_pulses)
        T = y.sum(axis=1)  # (n_trials,)

        pd_emp[idx] = np.mean(T > vt)

    return pd_emp


# -- Theoretical Pd helper ----------------------------------------------------
_theory_dispatch = {
    0: detection.pd_swerling0_nci,
    1: detection.pd_swerling1_nci,
    2: detection.pd_swerling2,
    3: detection.pd_swerling3_nci,
    4: detection.pd_swerling4,
}


# ── Figure 1: All five models, N = 10 ───────────────────────────────────
n_pulses_fig1 = 10
snr_db = np.linspace(-5, 25, 31)
snr_lin = 10 ** (snr_db / 10)

models = [
    (0, "Swerling 0", "o"),
    (2, "Swerling II", "s"),
    (4, "Swerling IV", "D"),
    (3, "Swerling III", "^"),
    (1, "Swerling I", "v"),
]

fig1, ax1 = plt.subplots(figsize=(9, 5))
for sw, label, marker in models:
    # Theory (smooth curve)
    pd_theory = _theory_dispatch[sw](snr_lin, pfa, n_pulses_fig1)
    ax1.plot(snr_db, pd_theory, label=f"{label} (theory)")

    # Empirical (markers)
    pd_emp = monte_carlo_pd(snr_db, n_pulses_fig1, sw, pfa, n_trials)
    ax1.plot(
        snr_db, pd_emp, marker, markersize=4, linestyle="none", alpha=0.7, label=f"{label} (MC)"
    )

ax1.set_xlabel("Per-pulse SNR [dB]")
ax1.set_ylabel("$P_d$")
ax1.set_title(
    f"Monte Carlo vs Theory — NCI $N = {n_pulses_fig1}$, "
    f"$P_{{fa}} = {pfa:.0e}$, {n_trials:,} trials"
)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(fontsize=7, ncol=2)
ax1.grid(True)
fig1.tight_layout()


# ── Figure 2: Convergence with N (Swerling I and III) ───────────────────
n_values = [1, 5, 10, 30]
snr_db2 = np.linspace(-5, 30, 25)
snr_lin2 = 10 ** (snr_db2 / 10)

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for ax, sw, sw_label, pd_func in [
    (ax2a, 1, "Swerling I", detection.pd_swerling1_nci),
    (ax2b, 3, "Swerling III", detection.pd_swerling3_nci),
]:
    for ci, n in enumerate(n_values):
        color = colors[ci % len(colors)]
        # Theory
        pd_th = pd_func(snr_lin2, pfa, n)
        ax.plot(snr_db2, pd_th, color=color, label=f"N={n} theory")
        # Empirical
        pd_emp = monte_carlo_pd(snr_db2, n, sw, pfa, n_trials)
        ax.plot(snr_db2, pd_emp, "o", color=color, markersize=3, alpha=0.6, label=f"N={n} MC")

    ax.set_xlabel("Per-pulse SNR [dB]")
    ax.set_ylabel("$P_d$")
    ax.set_title(f"{sw_label} — NCI convergence")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)

fig2.suptitle(f"Scan-to-Scan Models: NCI Gain vs N ($P_{{fa}} = {pfa:.0e}$, {n_trials:,} trials)")
fig2.tight_layout()


# ── Figure 3: False-alarm histogram (H₀ check) ─────────────────────────
n_pulses_h0 = 10
n_h0_trials = 200_000

noise_h0 = unity_variance_complex_noise((n_h0_trials, n_pulses_h0))
T_h0 = np.sum(np.abs(noise_h0) ** 2, axis=1)

vt = detection.threshold_factor_nci(pfa, n_pulses_h0)
empirical_pfa = np.mean(T_h0 > vt)

fig3, ax3 = plt.subplots(figsize=(8, 5))
# Histogram of test statistic
ax3.hist(T_h0, bins=100, density=True, alpha=0.6, label="Empirical $H_0$")

# Theoretical PDF: T = sum of N |CN(0,1)|^2 → each |CN(0,1)|^2 ~ Exp(1)
# so T ~ Gamma(N, 1) = chi2(2N) / 2
t_grid = np.linspace(0, T_h0.max(), 300)
pdf_theory = chi2.pdf(2 * t_grid, 2 * n_pulses_h0) * 2  # change of variable
ax3.plot(t_grid, pdf_theory, "r-", lw=2, label=r"$\chi^2(2N)/2$ PDF")

# Mark threshold
ax3.axvline(
    vt, color="k", ls="--", lw=1.5, label=f"$V_T$ (empirical $P_{{fa}}$ = {empirical_pfa:.1e})"
)

ax3.set_xlabel("NCI test statistic $T$")
ax3.set_ylabel("Density")
ax3.set_title(f"Noise-Only Distribution — $N = {n_pulses_h0}$, design $P_{{fa}} = {pfa:.0e}$")
ax3.legend()
ax3.grid(True)
fig3.tight_layout()


# ── Summary table ────────────────────────────────────────────────────────
snr_table_db = 10.0
snr_table_lin = 10 ** (snr_table_db / 10)
n_table = 10

print("=" * 60)
print("Monte Carlo Validation of Swerling NCI Detection Theory")
print("=" * 60)
print(f"Pfa = {pfa:.0e}, N = {n_table}, trials = {n_trials:,}")
print(f"SNR = {snr_table_db:.0f} dB (per pulse)")
print()
print(f"{'Model':<14} {'Theory':>8} {'Empirical':>10} {'Δ':>8}")
print("-" * 42)
for sw, label, _ in models:
    pd_th = float(_theory_dispatch[sw](snr_table_lin, pfa, n_table))
    pd_emp = monte_carlo_pd(np.array([snr_table_db]), n_table, sw, pfa, n_trials)[0]
    print(f"{label:<14} {pd_th:>8.4f} {pd_emp:>10.4f} {pd_emp - pd_th:>+8.4f}")

print()
print(
    f"False-alarm check: design Pfa = {pfa:.0e}, "
    f"empirical Pfa = {empirical_pfa:.2e} ({n_h0_trials:,} trials)"
)

plt.show()
