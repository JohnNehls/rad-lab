#!/usr/bin/env python
"""Swerling RCS fluctuation in the RDM simulation chain.

Demonstrates how Swerling target models affect a range-Doppler map when the
fluctuation is applied during datacube generation (not just in detection
theory).  Three identical targets at different ranges are assigned Swerling 0,
II, and IV models so the effect of pulse-to-pulse RCS variation is directly
visible in the RDM.

A second figure shows the same comparison for scan-to-scan models (I, III),
where the RCS is constant within a single dwell but varies between dwells.
Two independent dwells are generated side-by-side to show the inter-dwell
variation.

Key takeaways:
- Swerling 0 produces a clean, deterministic target peak in every dwell.
- Pulse-to-pulse models (II, IV) spread energy across Doppler sidelobes
  because the amplitude modulation from pulse to pulse acts like a random
  window, broadening the Doppler response.
- Scan-to-scan models (I, III) keep the Doppler response clean within a
  dwell, but the peak amplitude changes randomly between dwells.
- Higher DOF (III/IV vs I/II) produces less extreme amplitude swings
  because the RCS distribution is narrower.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

from rad_lab.pulse_doppler_radar import Radar
from rad_lab.waveform import lfm_waveform
from rad_lab.rf_datacube import data_cube, range_axis, matchfilter
from rad_lab.noise import unity_variance_complex_noise
from rad_lab._rdm_internals import add_skin, skin_snr_amplitude
from rad_lab.returns import Target

# -- Radar and waveform --
bw = 5e6
waveform = lfm_waveform(bw, T=1e-6, chirp_up_down=1)

radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (5 / 10),
    total_losses=10 ** (3 / 10),
    prf=10e3,
    dwell_time=6.4e-3,  # 64 pulses
)

waveform.set_sample(radar.sample_rate)


def make_rdm(targets, seed=0):
    """Build a datacube, inject targets + noise, match filter, Doppler FFT."""
    np.random.seed(seed)
    dc = data_cube(radar.sample_rate, radar.prf, radar.n_pulses)
    r_ax = range_axis(radar.sample_rate, dc.shape[0])

    for tgt in targets:
        amp = skin_snr_amplitude(radar, tgt, waveform)
        add_skin(dc, waveform, tgt, radar, amp)

    # Add noise
    noise = unity_variance_complex_noise(dc.shape) / np.sqrt(radar.n_pulses)
    dc += noise

    # Match filter
    matchfilter(dc, waveform.pulse_sample, pedantic=False)

    # Doppler FFT
    rdm = fft.fftshift(fft.fft(dc, axis=1), axes=1)
    f_axis = fft.fftshift(fft.fftfreq(dc.shape[1], 1 / radar.prf))
    rdot_axis = -3e8 * f_axis / (2 * radar.fcar)

    mag = np.abs(rdm)
    mag[mag == 0] = np.finfo(float).tiny
    db = 20 * np.log10(mag)
    return r_ax, rdot_axis, db


# -- Common target parameters --
avg_rcs = 10  # m^2
rdot = -200  # m/s

# ── Figure 1: Pulse-to-pulse models (0, II, IV) ─────────────────────────
targets_ptp = [
    Target(range=3e3, range_rate=rdot, rcs=avg_rcs, swerling=0),
    Target(range=5e3, range_rate=rdot, rcs=avg_rcs, swerling=2),
    Target(range=7e3, range_rate=rdot, rcs=avg_rcs, swerling=4),
]

r_ax, rdot_ax, db = make_rdm(targets_ptp, seed=42)
db_max = db.max()

fig1, ax1 = plt.subplots(figsize=(9, 5))
mesh = ax1.pcolormesh(rdot_ax * 1e-3, r_ax * 1e-3, db)
mesh.set_clim(db_max - 60, db_max)
fig1.colorbar(mesh, ax=ax1, label="dB")
ax1.set_xlabel("Range Rate [km/s]")
ax1.set_ylabel("Range [km]")
ax1.set_title("Pulse-to-Pulse RCS Fluctuation in RDM")

# Annotate targets
for tgt, label in zip(targets_ptp, ["Sw 0 (3 km)", "Sw II (5 km)", "Sw IV (7 km)"]):
    ax1.axhline(tgt.range * 1e-3, color="w", ls="--", lw=0.5, alpha=0.5)
    ax1.text(
        rdot_ax.min() * 1e-3 + 0.1,
        tgt.range * 1e-3 + 0.15,
        label,
        color="w",
        fontsize=9,
        fontweight="bold",
    )

fig1.tight_layout()

# ── Figure 2: Scan-to-scan models — two independent dwells ──────────────
targets_sts = [
    Target(range=3e3, range_rate=rdot, rcs=avg_rcs, swerling=0),
    Target(range=5e3, range_rate=rdot, rcs=avg_rcs, swerling=1),
    Target(range=7e3, range_rate=rdot, rcs=avg_rcs, swerling=3),
]

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

for ax, seed, title in [
    (ax2a, 10, "Dwell 1"),
    (ax2b, 20, "Dwell 2"),
]:
    r_ax, rdot_ax, db = make_rdm(targets_sts, seed=seed)
    db_max = db.max()
    mesh = ax.pcolormesh(rdot_ax * 1e-3, r_ax * 1e-3, db)
    mesh.set_clim(db_max - 60, db_max)
    fig2.colorbar(mesh, ax=ax, label="dB")
    ax.set_xlabel("Range Rate [km/s]")
    ax.set_ylabel("Range [km]")
    ax.set_title(title)
    for tgt, label in zip(targets_sts, ["Sw 0 (3 km)", "Sw I (5 km)", "Sw III (7 km)"]):
        ax.axhline(tgt.range * 1e-3, color="w", ls="--", lw=0.5, alpha=0.5)
        ax.text(
            rdot_ax.min() * 1e-3 + 0.1,
            tgt.range * 1e-3 + 0.15,
            label,
            color="w",
            fontsize=9,
            fontweight="bold",
        )

fig2.suptitle("Scan-to-Scan RCS Fluctuation: Same Targets, Different Dwells")
fig2.tight_layout()

# ── Figure 3: Doppler cuts at each target's range bin ────────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

# Pulse-to-pulse cuts
r_ax, rdot_ax, db = make_rdm(targets_ptp, seed=42)
for tgt, label, ls in zip(targets_ptp, ["Sw 0", "Sw II", "Sw IV"], ["-", "--", ":"]):
    r_idx = np.argmin(np.abs(r_ax - tgt.range))
    ax3a.plot(rdot_ax * 1e-3, db[r_idx, :], ls, label=label)

ax3a.set_xlabel("Range Rate [km/s]")
ax3a.set_ylabel("Magnitude [dB]")
ax3a.set_title("Doppler Cut — Pulse-to-Pulse Models")
ax3a.legend()
ax3a.grid(True)

# Scan-to-scan cuts (two dwells overlaid)
for seed, alpha in [(10, 1.0), (20, 0.5)]:
    r_ax, rdot_ax, db = make_rdm(targets_sts, seed=seed)
    dwell_label = f"dwell {seed}"
    for tgt, label, ls in zip(targets_sts, ["Sw 0", "Sw I", "Sw III"], ["-", "--", ":"]):
        r_idx = np.argmin(np.abs(r_ax - tgt.range))
        ax3b.plot(rdot_ax * 1e-3, db[r_idx, :], ls, alpha=alpha, label=f"{label} ({dwell_label})")

ax3b.set_xlabel("Range Rate [km/s]")
ax3b.set_ylabel("Magnitude [dB]")
ax3b.set_title("Doppler Cut — Scan-to-Scan Models (two dwells)")
ax3b.legend(fontsize=7, ncol=2)
ax3b.grid(True)
fig3.tight_layout()

# ── Summary ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Swerling RCS Fluctuation in RDM Simulation")
print("=" * 60)
print(
    f"Radar: fcar={radar.fcar / 1e9:.0f} GHz, PRF={radar.prf / 1e3:.0f} kHz, "
    f"N={radar.n_pulses} pulses"
)
print(f"Targets: avg RCS = {avg_rcs} m^2, range rate = {rdot} m/s")
print()
print("Pulse-to-pulse models (II, IV) modulate the amplitude independently")
print("on each pulse, spreading energy across Doppler sidelobes.")
print()
print("Scan-to-scan models (I, III) hold RCS constant within a dwell but")
print("vary it between dwells, causing amplitude differences across scans.")

plt.show()
