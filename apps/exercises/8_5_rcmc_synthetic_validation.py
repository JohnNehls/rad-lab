#!/usr/bin/env python
"""Validates RCMC against the closed-form migration formula.

Bypasses ``sar.gen`` and synthesises a perfectly range-compressed signal:
at each pulse ``eta``, the target's energy is a unit sinc lobe centred on
the analytic slant range ``R(eta) = sqrt(R0^2 + v^2 eta^2)``, modulated
by the two-way carrier phase ``exp(-j 4 pi R(eta) / lambda)``.

In the range-Doppler domain (azimuth FFT applied) the energy follows the
exact parabolic curve

    R(f_eta) = R0 / sqrt(1 - (lambda * f_eta / (2 v))^2)

which RCMC should straighten to a horizontal line at ``R0``.  The figure
overlays the measured trajectory before and after RCMC on this theory
curve, and shows the range-Doppler map in dB before and after correction.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

from rad_lab import SarRadar
from rad_lab import constants as c
from rad_lab._sar_internals import rcmc

# -- SAR geometry chosen so PRF >> Doppler bandwidth (no azimuth aliasing) --
sar_radar = SarRadar(
    fcar=10e9,
    tx_power=1,
    tx_gain=1,
    rx_gain=1,
    op_temp=290,
    sample_rate=10e6,
    noise_factor=1,
    total_losses=1,
    prf=20000,
    platform_velocity=400,
    aperture_length=500,
    platform_altitude=100,
)
R0 = 1000.0  # closest-approach range [m] — picked above the altitude

# -- Derived quantities --
n_pulses = sar_radar.n_pulses
n_range_bins = int(sar_radar.sample_rate / sar_radar.prf)
dR_grid = c.C / (2 * sar_radar.sample_rate)
range_axis = (np.arange(1, n_range_bins + 1)) * dR_grid
v = sar_radar.platform_velocity
lam = sar_radar.wavelength

doppler_bw = 2 * v * sar_radar.aperture_length / (R0 * lam)
peak_migration_m = sar_radar.aperture_length**2 / (8 * R0)
print(f"n_pulses={n_pulses}, n_range_bins={n_range_bins}, dR={dR_grid:.2f} m")
print(f"Doppler bandwidth = {doppler_bw:.0f} Hz, PRF = {sar_radar.prf} Hz")
print(
    f"Predicted peak migration: {peak_migration_m:.2f} m "
    f"({peak_migration_m / dR_grid:.2f} range cells)"
)

# -- Synthesise a perfectly range-compressed point-target signal --
# Slow-time axis centred so eta=0 is broadside (closest approach).
eta = (np.arange(n_pulses) - n_pulses // 2) / sar_radar.prf
R_eta = np.sqrt(R0**2 + (v * eta) ** 2)  # exact hyperbolic range history
phase = -4 * np.pi / lam * R_eta  # two-way carrier phase

datacube = np.zeros((n_range_bins, n_pulses), dtype=np.complex64)
for m in range(n_pulses):
    sinc_envelope = np.sinc((range_axis - R_eta[m]) / dR_grid)
    datacube[:, m] = sinc_envelope * np.exp(1j * phase[m])

# -- RDM before RCMC --
rdm_before = fft.fftshift(fft.fft(datacube, axis=1), axes=1)
f_eta = fft.fftshift(fft.fftfreq(n_pulses, d=1.0 / sar_radar.prf))

k_target = int(np.round(R0 / dR_grid - 1))
search = slice(max(0, k_target - 3), min(n_range_bins, k_target + 10))
peaks_before = np.argmax(np.abs(rdm_before[search, :]), axis=0) + search.start

# -- Apply RCMC, then recompute the RDM --
rcmc(datacube, sar_radar, range_axis)
rdm_after = fft.fftshift(fft.fft(datacube, axis=1), axes=1)
peaks_after = np.argmax(np.abs(rdm_after[search, :]), axis=0) + search.start

# -- Closed-form migration curve for overlay --
arg = np.clip(lam * f_eta / (2 * v), -0.999, 0.999)
theory_R = R0 / np.sqrt(1 - arg**2)

# -- Plot --
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(f_eta, range_axis[peaks_before], "b-", linewidth=2, label="before RCMC")
ax.plot(f_eta, theory_R, "g--", linewidth=2, label="theory R0/√(1−(λfη/2v)²)")
ax.plot(f_eta, range_axis[peaks_after], "r-", linewidth=2, label="after RCMC")
ax.axhline(R0, color="k", linestyle=":", label=f"R0 = {R0:.0f} m")
ax.set_xlabel("Azimuth Doppler frequency [Hz]")
ax.set_ylabel("Peak range [m]")
ax.set_title("Peak range vs Doppler")
ax.legend()
ax.grid(True)

zoom = slice(max(0, k_target - 5), min(n_range_bins, k_target + 12))
for ax_idx, (rdm, title) in enumerate(
    [(rdm_before, "RDM before RCMC (dB)"), (rdm_after, "RDM after RCMC (dB)")], start=1
):
    ax = axes[ax_idx]
    mag = np.abs(rdm[zoom, :])
    db = 20 * np.log10(mag / mag.max() + 1e-30)
    mesh = ax.pcolormesh(f_eta, range_axis[zoom], db, vmin=-30, vmax=0, shading="auto")
    ax.axhline(R0, color="r", linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Azimuth Doppler frequency [Hz]")
    ax.set_ylabel("Range [m]")
    fig.colorbar(mesh, ax=ax, label="Magnitude [dB]")

fig.tight_layout()
plt.show()
