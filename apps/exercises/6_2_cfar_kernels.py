#!/usr/bin/env python
"""CFAR kernels in 1-D: how CA, GOCA, and SOCA differ.

The applied CFAR exercise (6_3) runs on a 2-D range-Doppler map, where the
detection threshold is a surface you cannot see directly.  Strip the problem
down to a single range profile so the threshold is a *line* drawn on top of
the signal, making the kernel arithmetic visible:

Each cell under test (CUT) estimates the local noise from N training cells
on each side (a guard band around the CUT is excluded).  The three kernels
combine the leading and lagging halves differently:

    CA    noise = mean(all 2N training cells)
    GOCA  noise = max(mean(leading half), mean(lagging half))
    SOCA  noise = min(mean(leading half), mean(lagging half))

The threshold is ``alpha * noise`` with the same CA multiplier used in
`cfar_2d`.  That multiplier is exact only for CA; GOCA
and SOCA reuse it as an approximation (their exact multipliers have no
closed form), so their realised false-alarm rate drifts from ``Pfa``.

The profile contains two features that expose the trade-offs:

1. Two closely-spaced targets, near enough that each sits in the other's
   training window.
2. A clutter edge — a step up in noise power.

Key takeaways:
- Target masking: the strong target inflates one half of the weaker
  target's window.  CA averages it in and GOCA keys on that inflated half,
  raising the threshold until the weak target is MASKED.  SOCA keys on the
  clean half and still detects it.
- Clutter edge: SOCA clings to the low (clear) side as the window crosses
  the edge, so its threshold lags and it fires false alarms just inside the
  clutter.  GOCA keys on the high side and stays clean; CA ramps between.
- No kernel wins everywhere: SOCA resists mutual-target masking but is worst
  at clutter edges; GOCA is the reverse; CA is the compromise.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

N = 200  # range cells
N_GUARD = 2  # guard cells per side
N_TRAIN = 10  # training cells per side
PFA = 1e-3
EDGE = 130  # clutter step location [cell]


def cfar_1d(power, method):
    """1-D CFAR threshold at every cell.

    Cells too close to either end for a full window are left as NaN.

    Args:
        power: 1-D array of cell powers (square-law detector output).
        method: ``"CA"``, ``"GOCA"``, or ``"SOCA"``.

    Returns:
        Threshold power at each cell, same shape as *power*.
    """
    n = len(power)
    thr = np.full(n, np.nan)
    half = N_GUARD + N_TRAIN
    n_train = 2 * N_TRAIN
    # Exact CA multiplier for exponential noise: alpha = N*(Pfa^(-1/N) - 1)
    alpha = n_train * (PFA ** (-1.0 / n_train) - 1)
    for i in range(half, n - half):
        lead = power[i - half : i - N_GUARD]  # N_TRAIN cells before the guard band
        lag = power[i + N_GUARD + 1 : i + half + 1]  # N_TRAIN cells after the guard band
        if method == "CA":
            noise = np.mean(np.concatenate([lead, lag]))
        elif method == "GOCA":
            noise = max(lead.mean(), lag.mean())
        else:  # SOCA
            noise = min(lead.mean(), lag.mean())
        thr[i] = alpha * noise
    return thr


# -- Build the range profile: exponential noise, a clutter step, two targets --
mean_power = np.ones(N)
mean_power[EDGE:] = 30.0  # clutter step: ~15 dB above the clear-region floor
power = rng.exponential(mean_power)

# Two closely-spaced targets in the clear region (6 cells apart, inside the
# +/-12-cell window): the strong one contaminates one half of the weak one's
# training window.
power[45] = 400.0  # strong target, ~26 dB
power[51] = 120.0  # weak target, ~21 dB -- the one at risk of masking

# -- One stacked panel per kernel: signal, its threshold, and its detections --
colors = {"CA": "tab:blue", "GOCA": "tab:green", "SOCA": "tab:red"}
signal_db = 10 * np.log10(power)

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, sharey=True)
fig.suptitle(f"1-D CFAR kernels (Pfa={PFA:.0e}, {N_TRAIN} training + {N_GUARD} guard per side)")

print("## 1-D CFAR kernels ##")
for ax, (method, c) in zip(axes, colors.items()):
    thr = cfar_1d(power, method)
    hits = np.where(power > thr)[0]
    n_fa = int((power[EDGE:] > thr[EDGE:]).sum())
    print(
        f"  {method}: target@45 {power[45] > thr[45]}, target@51 {power[51] > thr[51]}, "
        f"{n_fa} false alarms inside clutter"
    )

    ax.axvspan(EDGE, N, color="0.9", zorder=0)  # clutter region
    for cell in (45, 51):  # faint guides at the two target cells
        ax.axvline(cell, color="0.8", linewidth=0.8, zorder=0)
    ax.plot(signal_db, color="0.55", linewidth=0.8)
    ax.plot(10 * np.log10(thr), color=c, linewidth=1.8)
    ax.plot(
        hits,
        signal_db[hits],
        "o",
        color=c,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.6,
    )

    ax.set_ylabel("Power [dB]")
    # Name the kernel in-panel (colored) instead of a repeated legend
    ax.text(0.01, 0.9, method, transform=ax.transAxes, color=c, fontweight="bold", va="top")

# Label the shared features once, on the top panel only
top = axes[0]
top.plot([], [], color="0.55", linewidth=0.8, label="signal")
top.plot([], [], "ko", markersize=5, label="detections")
top.legend(loc="upper right", framealpha=0.9)
top.annotate("clutter", (EDGE + 3, 34), fontsize=9, color="0.4", va="top")
# The two targets are only 6 cells apart; the faint guides mark them, so one
# centered label avoids overlapping text
top.annotate("targets (45, 51)", (48, 34), fontsize=8, ha="center", va="top")

axes[-1].set_xlabel("Range cell")
axes[0].set_ylim(-15, 38)
fig.tight_layout()

plt.show()
