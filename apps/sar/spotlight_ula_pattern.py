#!/usr/bin/env python
"""Spotlight SAR with a ULA-derived beam pattern.

Replaces the default Gaussian beam pattern with a realistic array factor
computed from a 20-element uniform linear array (λ/2 spacing).  The ULA
pattern introduces sidelobes and nulls that affect how targets at different
off-boresight angles are weighted.

Compares Gaussian vs ULA beam patterns by running the same scene twice.
"""

import matplotlib.pyplot as plt
import numpy as np
import rad_lab.uniform_linear_arrays as ula
from rad_lab import sar, SarRadar, SarTarget, lfm_waveform

# -- Waveform --
bw = 5e6
waveform = lfm_waveform(bw, T=10e-6, chirp_up_down=1)

# -- ULA beam pattern --
# 20 elements at λ/2 spacing → array length = 9.5λ → beamwidth ≈ 5.1°
n_elements = 20
dx = 0.5  # element spacing in wavelengths
L = (n_elements - 1) * dx
el_pos = np.linspace(-L / 2, L / 2, n_elements)

pattern_fn = ula.ula_pattern(el_pos, two_way=True)

# -- Show both patterns --
theta_rad = np.linspace(0, np.radians(15), 500)
gauss_weights = np.exp(-4 * np.log(2) * (theta_rad / np.radians(5.1)) ** 2)
ula_weights = pattern_fn(theta_rad)

fig_pat, ax_pat = plt.subplots(figsize=(8, 4))
ax_pat.plot(np.degrees(theta_rad), 10 * np.log10(gauss_weights + 1e-30), label="Gaussian")
ax_pat.plot(np.degrees(theta_rad), 10 * np.log10(ula_weights + 1e-30), label="ULA (20 el)")
ax_pat.set_xlabel("Off-boresight angle [deg]")
ax_pat.set_ylabel("Two-way amplitude weight [dB]")
ax_pat.set_ylim(-40, 3)
ax_pat.legend()
ax_pat.grid(True)
ax_pat.set_title("Gaussian vs ULA beam pattern")
fig_pat.tight_layout()

# -- SAR parameters (spotlight mode) --
scene_center = [0, 3e3, 0]
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
    aperture_length=200,
    platform_altitude=5e3,
    scene_center=scene_center,
    beamwidth=np.radians(5.1),
)

sar_radar = SarRadar(**common)

targets = [
    SarTarget(position=[-5, 0.1e3, 0], rcs=10),
    SarTarget(position=[0, 3e3, 0], rcs=10),
    SarTarget(position=[5, 5e3, 0], rcs=10),
]

# -- Run with default Gaussian --
print("Running spotlight with Gaussian beam pattern ...")
cr_g, sr_g, img_g = sar.gen(sar_radar, waveform, targets, seed=0, plot=False)

# -- Run with ULA pattern --
print("Running spotlight with ULA beam pattern ...")
cr_u, sr_u, img_u = sar.gen(
    sar_radar, waveform, targets, seed=0, plot=False, beam_pattern=pattern_fn
)

# -- Compare focused images --
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, cr, sr_ax, img, title in [
    (axes[0], cr_g, sr_g, img_g, "Gaussian beam"),
    (axes[1], cr_u, sr_u, img_u, "ULA beam (20 el)"),
]:
    mag = np.abs(img)
    mag_db = 20 * np.log10(mag / mag.max() + 1e-30)
    mesh = ax.pcolormesh(cr, sr_ax / 1e3, mag_db, vmin=-40, vmax=0)
    ax.set_title(title)
    ax.set_xlabel("Cross-Range [m]")
    ax.set_ylabel("Slant Range [km]")
    fig.colorbar(mesh, ax=ax, label="dB")

fig.suptitle("Spotlight SAR: Gaussian vs ULA beam pattern", fontsize=14)
fig.tight_layout()

plt.show()
