#!/usr/bin/env python
"""LFM range-sidelobe suppression by matched-filter weighting.

The problem:
  - An LFM (chirp) compresses to a sinc-like range response with peak
    sidelobes only ~-13.2 dB below the mainlobe.
  - CFAR thresholds off the local noise floor and knows nothing of pulse
    shape, so around a strong target it fires on those sidelobes -- a smear
    of spurious "targets" where there is really one.
  - CFAR cannot fix this itself: a -13 dB sidelobe *is* a real power excess.

The fix (one stage earlier, in pulse compression): weight the matched-filter
replica.  An LFM's frequency is linear in time, so tapering the replica in
time tapers the swept-spectrum edges -- the frequency weighting Richards uses
for range-sidelobe control.  Sidelobes drop below threshold; the cost is a
broader mainlobe (coarser range resolution) and a small SNR loss.

Produces two figures:

- **Fig 1:** noiseless range point-spread (rectangular vs Taylor vs Chebyshev)
  — the -13.2 dB pedestal buried, the mainlobe widened.
- **Fig 2:** CA-CFAR on the noisy RDM — the spurious sidelobe detections
  vanish.

Printed per window: peak sidelobe level, -3 dB mainlobe width, weighting loss,
CFAR detection count.

Reference: Richards, M. A., *Fundamentals of Radar Signal Processing*, 2nd
ed., McGraw-Hill, 2014, Ch. 4 (Radar Waveforms) -- matched filter, LFM pulse
compression, the ~-13.2 dB sidelobes, and their reduction by amplitude
weighting with the attendant processing-gain loss.  CFAR is Ch. 7.
"""

from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from rad_lab import rdm
from rad_lab.pulse_doppler_radar import Radar
from rad_lab.waveform import lfm_waveform
from rad_lab.returns import Target, Return
from rad_lab.rf_datacube import range_window
from rad_lab.cfar import cfar_2d

# Windows to compare.  "none" is the plain matched filter; scipy's taylor()
# takes sll as a positive suppression in dB (sidelobe level is -sll).
WINDOWS = [
    ("none", None, "Rectangular (no weighting)"),
    ("taylor", {"nbar": 5, "sll": 35}, "Taylor (nbar=5, -35 dB)"),
    ("chebyshev", {"at": 60}, "Chebyshev (-60 dB)"),
]

# 1 us LFM at 40 MHz (vs 10 MHz in the CFAR exercises): the larger TB = T*B = 40
# lets time-domain replica weighting reach the windows' design sidelobe levels.
bw = 40e6  # Hz
waveform = lfm_waveform(bw, T=1.0e-6, chirp_up_down=1)

# -- Radar --
radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=200e3,
    dwell_time=2e-3,
)

# Single strong target, RCS tuned so the ~-13 dB unweighted sidelobes clear the
# noise floor (CFAR detects them) but the ~-32 dB weighted sidelobes do not.
target = Target(range=5.0e3, range_rate=300.0, rcs=30.0)
return_list = [Return(target=target)]


def range_profile_db(dc, rdot_axis, range_rate):
    """Range cut (dB, peak-normalised) through the target's Doppler column."""
    d_col = np.argmin(np.abs(rdot_axis - range_rate))
    profile = np.abs(dc[:, d_col])
    profile = profile / profile.max()
    return 20 * np.log10(np.maximum(profile, 1e-12))


def mainlobe_nulls(profile_db, peak_idx):
    """Indices of the first null on each side of the mainlobe peak.

    Walk outward from the peak while the profile keeps descending; the first
    sample that stops descending is the null bounding the mainlobe.
    """

    def walk(direction):
        i = peak_idx
        while 0 < i < len(profile_db) - 1 and profile_db[i + direction] < profile_db[i]:
            i += direction
        return i

    return walk(-1), walk(1)


def peak_sidelobe_db(profile_db, peak_idx):
    """Peak range sidelobe level [dB] relative to the mainlobe peak."""
    lo, hi = mainlobe_nulls(profile_db, peak_idx)
    sidelobes = np.concatenate([profile_db[: lo + 1], profile_db[hi:]])
    return sidelobes.max()


def mainlobe_width_bins(profile_db, peak_idx):
    """-3 dB mainlobe width in range bins (linear interp between samples)."""

    def cross(direction):
        i = peak_idx
        while 0 < i < len(profile_db) - 1 and profile_db[i] > -3.0:
            i += direction
        # linear interpolate the -3 dB crossing between i-direction and i
        y0, y1 = profile_db[i - direction], profile_db[i]
        frac = (-3.0 - y0) / (y1 - y0)
        return (i - direction) + frac * direction

    return cross(1) - cross(-1)


def weighting_loss_db(n_taps, window, window_kwargs):
    """Matched-filter (weighting) SNR loss of a windowed replica [dB].

    For a unit-amplitude replica ``p`` (|p|=1) weighted by ``w``, the SNR
    relative to the matched (rectangular) filter is the mismatch ratio
    ``|sum w|^2 / (N * sum w^2)`` (Cauchy-Schwarz: 1 only when w is flat).
    """
    w = range_window(n_taps, window, window_kwargs)
    ratio = np.abs(w.sum()) ** 2 / (n_taps * np.sum(w**2))
    return -10 * np.log10(ratio)  # report loss as a positive number of dB


# =====================================================================
# Fig 1: noiseless range point-spread function
# =====================================================================
print("## Range point-spread function (noiseless) ##")
radar_noiseless = replace(radar, op_temp=0)  # thermal noise scales with temp

dR = 3e8 / (2 * radar.sample_rate)  # range-bin spacing [m]

fig1, ax1 = plt.subplots(figsize=(9, 5))
fig1.suptitle("LFM range sidelobes vs matched-filter weighting")

for win, kwargs, label in WINDOWS:
    rdot_axis, r_axis, dc = rdm.gen(
        radar_noiseless,
        waveform,
        return_list,
        plot=False,
        range_window=win,
        range_window_kwargs=kwargs,
    )
    prof = range_profile_db(dc, rdot_axis, target.range_rate)
    peak = int(np.argmax(prof))

    psl = peak_sidelobe_db(prof, peak)
    width = mainlobe_width_bins(prof, peak)
    # pulse_sample is populated by rdm.gen -> waveform.set_sample()
    n_taps = waveform.pulse_sample.size
    loss = weighting_loss_db(n_taps, win, kwargs)
    print(
        f"  {label:32s}  PSL={psl:6.1f} dB   "
        f"-3dB width={width * dR:5.1f} m ({width:.2f} bins)   "
        f"weighting loss={loss:4.2f} dB"
    )

    # Plot a window of range bins around the target so the sidelobes are visible
    r_km = r_axis * 1e-3
    sel = slice(max(peak - 25, 0), peak + 26)
    ax1.plot(r_km[sel], prof[sel], marker=".", markersize=3, label=label)

ax1.axhline(-13.2, color="k", ls=":", lw=1, label="-13.2 dB (rect. LFM sidelobe)")
ax1.set_ylim(-80, 3)
ax1.set_xlabel("Range [km]")
ax1.set_ylabel("Normalised magnitude [dB]")
ax1.set_title(f"Target at {target.range * 1e-3:.1f} km, {target.range_rate:.0f} m/s")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()


# =====================================================================
# Fig 2: CA-CFAR on the noisy RDM -- spurious sidelobe detections vanish
# =====================================================================
PFA = 1e-6
CFAR_PARAMS = dict(
    n_guard_range=2,
    n_guard_doppler=2,
    n_train_range=8,
    n_train_doppler=8,
    pfa=PFA,
)

R_UNAMB = 3e8 / (2 * radar.prf)  # unambiguous range [m]; the 5 km target aliases in
MAINLOBE_HALF = 4  # cells: a box this size around the target counts as "the target"

print(f"\n## CA-CFAR on the RDM (Pfa={PFA:.0e}) ##")
print("  (spurious = detected cells outside a mainlobe box around the true target)")
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle(f"CA-CFAR vs range weighting (Pfa={PFA:.0e})")

for ax, (win, kwargs, label) in zip(axes2, WINDOWS):
    rdot_axis, r_axis, dc = rdm.gen(
        radar,
        waveform,
        return_list,
        plot=False,
        range_window=win,
        range_window_kwargs=kwargs,
    )
    dets, _ = cfar_2d(dc, method="CA", **CFAR_PARAMS)

    # There is only one target, so every detection outside a small box around
    # it is a range-sidelobe false alarm.
    r_row = np.argmin(np.abs(r_axis - target.range % R_UNAMB))
    d_col = np.argmin(np.abs(rdot_axis - target.range_rate))
    dr = np.abs(np.arange(dc.shape[0])[:, None] - r_row)
    dd = np.abs(np.arange(dc.shape[1])[None, :] - d_col)
    mainlobe_box = (dr <= MAINLOBE_HALF) & (dd <= MAINLOBE_HALF)

    n_det = int(dets.sum())
    n_spurious = int((dets & ~mainlobe_box).sum())
    print(f"  {label:32s}  {n_det:3d} total detections, {n_spurious:2d} spurious sidelobe cells")

    magnitude = np.abs(dc)
    magnitude[magnitude == 0] = np.finfo(float).tiny
    plot_data = 20 * np.log10(magnitude / magnitude.max())
    mesh = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, plot_data, shading="auto")
    mesh.set_clim(-60, 0)

    det_r, det_d = np.where(dets)
    if det_r.size:
        ax.plot(
            rdot_axis[det_d] * 1e-3,
            r_axis[det_r] * 1e-3,
            "rx",
            markersize=5,
            label="CFAR detections",
        )
    ax.set_title(f"{label}\n({n_spurious} spurious sidelobe detections)")
    ax.set_xlabel("Range Rate [km/s]")
    ax.set_ylabel("Range [km]")

axes2[-1].legend(loc="upper right")
fig2.subplots_adjust(top=0.80)  # room for the two-line panel titles under the suptitle
fig2.colorbar(mesh, ax=axes2, label="Normalised Magnitude [dB]")

print(
    "\nTakeaway: weighting the replica sinks the LFM range sidelobes below the\n"
    "CFAR threshold, so the spurious sidelobe detections vanish -- at the cost\n"
    "of ~1 dB SNR and a wider mainlobe (the mainlobe cluster even grows)."
)

plt.show()
