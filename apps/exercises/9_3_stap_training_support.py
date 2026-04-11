#!/usr/bin/env python
"""STAP training support: the Reed-Mallett-Brennan (RMB) rule.

Sample Matrix Inversion estimates the clutter covariance from ``K`` training
snapshots (secondary range bins).  With too few, the adaptive weights are
mismatched and SINR is lost.  Reed, Mallett & Brennan (1974) showed the
*average* SINR loss relative to the clairvoyant optimum depends only on ``K``
and the number of space-time degrees of freedom ``NM``:

    E[ SINR(w_hat) / SINR_opt ]  =  (K + 2 - NM) / (K + 1)

The practical consequence is the "2 NM rule": using ``K = 2 NM`` training
snapshots costs ~3 dB of SINR, and this is roughly independent of the clutter
scenario.  It is why full-dimension STAP is data-hungry (here NM = 128, so
2 NM = 256 homogeneous training bins) and why reduced-dimension methods exist.

This exercise draws synthetic training snapshots from the theoretical
clutter-plus-noise covariance, forms the SMI weights, and compares the measured
SINR loss against the RMB prediction across a sweep of ``K``.
"""

import numpy as np
import matplotlib.pyplot as plt

from rad_lab import stap
from rad_lab.pulse_doppler_radar import Radar, frequency_delta_doppler

rng = np.random.default_rng(0)

# -- Radar / array (same platform as the other STAP exercises) --
bw = 1e6
radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (20 / 10),
    rx_gain=10 ** (20 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (5 / 10),
    total_losses=10 ** (3 / 10),
    prf=10e3,
    dwell_time=1.6e-3,  # 16 pulses
)

n_elements = 8
el_pos = np.arange(n_elements) * 0.5 - (n_elements - 1) * 0.25
n_pulses = radar.n_pulses
NM = n_elements * n_pulses  # space-time degrees of freedom

platform_velocity = 100  # m/s
cnr = 40

# -- Theoretical clutter-plus-noise covariance and its Cholesky factor --
# Snapshots are drawn as x = L z with z ~ CN(0, I), giving cov(x) = L L^H = R.
R = stap.clutter_plus_noise_covariance(el_pos, n_pulses, radar, platform_velocity, cnr=cnr)
R_inv = np.linalg.inv(R)
L_chol = np.linalg.cholesky(R)

# -- Target steering: broadside, well off the clutter ridge (rdot = -50 m/s) --
fd_target = frequency_delta_doppler(-50, radar.fcar)
s = stap.space_time_steering_vector(el_pos, n_pulses, 0.0, fd_target, radar.prf)
sinr_opt = np.real(np.vdot(s, R_inv @ s))  # clairvoyant optimum

print("## STAP Training Support (RMB rule) ##")
print(f"  Degrees of freedom NM = {NM}")
print(f"  Optimal (clairvoyant) SINR = {10 * np.log10(sinr_opt):.1f} dB")
print(f"  {'K':>5} {'K/NM':>6} {'sim loss':>10} {'RMB theory':>11}")

# -- Sweep the number of training snapshots K --
K_values = np.unique(np.linspace(NM, 4 * NM, 14).astype(int))
n_trials = 60

sim_loss_db = np.zeros(len(K_values))
rmb_loss_db = np.zeros(len(K_values))

for i, K in enumerate(K_values):
    losses = np.zeros(n_trials)
    for t in range(n_trials):
        # K training snapshots with covariance R
        z = (rng.standard_normal((NM, K)) + 1j * rng.standard_normal((NM, K))) / np.sqrt(2)
        x = L_chol @ z
        R_hat = (x @ x.conj().T) / K

        # SMI weights from the estimate, evaluated against the true covariance
        w = np.linalg.solve(R_hat, s)
        sinr = np.abs(np.vdot(w, s)) ** 2 / np.real(np.vdot(w, R @ w))
        losses[t] = sinr / sinr_opt

    sim_loss_db[i] = 10 * np.log10(losses.mean())
    rmb_loss_db[i] = 10 * np.log10((K + 2 - NM) / (K + 1))
    print(f"  {K:5d} {K / NM:6.2f} {sim_loss_db[i]:9.1f} dB {rmb_loss_db[i]:10.1f} dB")

# -- Plot --
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_values / NM, sim_loss_db, "o", label=f"simulated ({n_trials} trials)")
ax.plot(K_values / NM, rmb_loss_db, "-", label="RMB rule  $(K{+}2{-}NM)/(K{+}1)$")
ax.axhline(-3, color="gray", ls=":", label="-3 dB")
ax.axvline(2, color="gray", ls="--", label="$K = 2NM$")
ax.set_title(f"STAP SMI training loss vs support (NM = {NM})")
ax.set_xlabel("Training snapshots  $K / NM$")
ax.set_ylabel("Average SINR loss [dB]")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

plt.show()
