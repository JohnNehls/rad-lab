"""Multiple-PRF ambiguity resolution.

Provides functions to resolve range and range-rate ambiguities by comparing
detections across CPIs collected at different pulse repetition frequencies.

A single PRF creates ambiguity windows in both range
(:math:`R_{ua} = c / 2 \\cdot \\text{PRF}`) and range-rate
(:math:`\\dot{r}_{ua} = \\pm c \\cdot \\text{PRF} / (4 f_c)`).
Targets beyond these limits alias back into the window.  By processing the
same scene at two or more PRFs whose unambiguous intervals differ, the true
target parameters can be recovered: the correct value is the one that aliases
consistently into the observed bin at every PRF.

Reference: Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., 2014,
§3.7 — "Multiple PRF Ranging and Velocity Ambiguity Resolution".
"""

import numpy as np
import matplotlib.pyplot as plt

from .pulse_doppler_radar import range_unambiguous, range_rate_pm_unambiguous


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Combined unambiguous intervals
# ---------------------------------------------------------------------------


def combined_unambiguous_range(prfs):
    """Approximate combined unambiguous range for a set of PRFs.

    The combined unambiguous range is the least common multiple of the
    individual unambiguous ranges.  Because the individual ranges are
    real-valued (not integers), this function discretises to 1-metre
    resolution before computing the LCM.

    Args:
        prfs: Sequence of PRF values [Hz].

    Returns:
        Combined unambiguous range [m].
    """
    from math import gcd

    r_uas = [int(round(range_unambiguous(p))) for p in prfs]
    lcm = r_uas[0]
    for r in r_uas[1:]:
        lcm = lcm * r // gcd(lcm, r)
    return float(lcm)


def combined_unambiguous_range_rate(prfs, fcar):
    """Approximate combined unambiguous range-rate for a set of PRFs.

    Discretises the individual unambiguous range-rate limits to 0.01 m/s
    resolution before computing the LCM.

    Args:
        prfs: Sequence of PRF values [Hz].
        fcar: Carrier frequency [Hz].

    Returns:
        Combined unambiguous range-rate magnitude [m/s].  The full
        interval is :math:`\\pm` this value.
    """
    from math import gcd

    # Work in centimetres-per-second to get integers
    rdot_uas = [int(round(range_rate_pm_unambiguous(p, fcar) * 100)) for p in prfs]
    lcm = rdot_uas[0]
    for r in rdot_uas[1:]:
        lcm = lcm * r // gcd(lcm, r)
    return lcm / 100.0


# ---------------------------------------------------------------------------
# Coincidence resolvers
# ---------------------------------------------------------------------------


def resolve_range(observed_ranges, prfs, tolerance, max_range=None, step=None):
    """Find true target ranges consistent with aliased observations at multiple PRFs.

    For each candidate true range on a search grid, the function checks
    whether its aliased position at every PRF falls within *tolerance* of
    the corresponding observed range.

    Args:
        observed_ranges: Length-N sequence of observed (aliased) ranges [m],
            one per PRF.
        prfs: Length-N sequence of PRFs [Hz].
        tolerance: Maximum allowed discrepancy [m] between the candidate's
            aliased range and the observation at each PRF.
        max_range: Upper bound of the search grid [m].  Defaults to the
            combined unambiguous range.
        step: Search grid spacing [m].  Defaults to ``tolerance / 2`` for a
            fine-enough grid.

    Returns:
        1-D array of candidate true ranges [m] that match all observations.
    """
    if max_range is None:
        max_range = combined_unambiguous_range(prfs)
    if step is None:
        step = tolerance / 2

    candidates = np.arange(0, max_range, step)
    mask = np.ones(len(candidates), dtype=bool)

    for r_obs, prf in zip(observed_ranges, prfs):
        r_ua = range_unambiguous(prf)
        aliased = candidates % r_ua
        mask &= np.abs(aliased - r_obs) < tolerance
    return candidates[mask]


def resolve_range_rate(observed_rdots, prfs, fcar, tolerance, max_rdot=None, step=None):
    """Find true target range-rates consistent with aliased observations at multiple PRFs.

    Args:
        observed_rdots: Length-N sequence of observed (aliased) range-rates
            [m/s], one per PRF.
        prfs: Length-N sequence of PRFs [Hz].
        fcar: Carrier frequency [Hz].
        tolerance: Maximum allowed discrepancy [m/s].
        max_rdot: Upper bound of the search magnitude [m/s].  Defaults to
            the combined unambiguous range-rate.
        step: Search grid spacing [m/s].  Defaults to ``tolerance / 2``.

    Returns:
        1-D array of candidate true range-rates [m/s].
    """
    if max_rdot is None:
        max_rdot = combined_unambiguous_range_rate(prfs, fcar)
    if step is None:
        step = tolerance / 2

    candidates = np.arange(-max_rdot, max_rdot, step)
    mask = np.ones(len(candidates), dtype=bool)

    for rdot_obs, prf in zip(observed_rdots, prfs):
        rdot_ua = range_rate_pm_unambiguous(prf, fcar)
        aliased = (candidates + rdot_ua) % (2 * rdot_ua) - rdot_ua
        mask &= np.abs(aliased - rdot_obs) < tolerance
    return candidates[mask]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_multiprf_rdms(rdot_axes, r_axes, rdms, prfs, title="Multiple-PRF RDMs"):
    """Plot RDMs from multiple PRFs side by side.

    Args:
        rdot_axes: List of 1-D range-rate axes [m/s], one per PRF.
        r_axes: List of 1-D range axes [m], one per PRF.
        rdms: List of 2-D complex RDM arrays, one per PRF.
        prfs: Sequence of PRF values [Hz] (for subplot titles).
        title: Overall figure title.

    Returns:
        (fig, axes) tuple.
    """
    n = len(prfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes.ravel()

    for i, (rdot, r, rdm, prf) in enumerate(zip(rdot_axes, r_axes, rdms, prfs)):
        mag = np.abs(rdm)
        mag[mag == 0] = np.finfo(float).tiny
        db = 20 * np.log10(mag)
        db_max = db.max()

        mesh = axes[i].pcolormesh(rdot * 1e-3, r * 1e-3, db)
        mesh.set_clim(db_max - 60, db_max)
        axes[i].set_xlabel("Range Rate [km/s]")
        axes[i].set_ylabel("Range [km]")
        axes[i].set_title(f"PRF = {prf / 1e3:.1f} kHz")
        fig.colorbar(mesh, ax=axes[i], label="dB")

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_resolved(
    true_ranges,
    true_rdots,
    resolved_ranges,
    resolved_rdots,
    prfs,
    fcar,
    title="Ambiguity Resolution",
):
    """Visualise how aliased detections at each PRF map to a single true value.

    Produces two subplots (range and range-rate).  For each PRF the
    unambiguous interval is drawn as a shaded band and the observed
    (aliased) detection is marked.  The resolved true value is shown as
    a vertical dashed line.

    Args:
        true_ranges: Sequence of observed aliased ranges [m], one per PRF.
        true_rdots: Sequence of observed aliased range-rates [m/s], one per PRF.
        resolved_ranges: Array of resolved true-range candidates [m]
            (typically a single value).
        resolved_rdots: Array of resolved true range-rate candidates [m/s].
        prfs: PRF values [Hz].
        fcar: Carrier frequency [Hz].
        title: Figure title.

    Returns:
        (fig, axes) tuple.
    """
    n = len(prfs)
    fig, (ax_r, ax_v) = plt.subplots(2, 1, figsize=(8, 6))

    prf_labels = [f"{p / 1e3:.1f} kHz" for p in prfs]
    y_pos = np.arange(n)

    # --- Range subplot ---
    for i, prf in enumerate(prfs):
        r_ua = range_unambiguous(prf)
        ax_r.barh(i, r_ua / 1e3, height=0.4, alpha=0.2, color="C0")
        ax_r.plot(true_ranges[i] / 1e3, i, "ko", markersize=8)

    if len(resolved_ranges) > 0:
        ax_r.axvline(resolved_ranges[0] / 1e3, color="r", ls="--", label="Resolved range")
        ax_r.legend()

    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(prf_labels)
    ax_r.set_xlabel("Range [km]")
    ax_r.set_title("Range Ambiguity Resolution")

    # --- Range-rate subplot ---
    for i, prf in enumerate(prfs):
        rdot_ua = range_rate_pm_unambiguous(prf, fcar)
        ax_v.barh(i, 2 * rdot_ua, left=-rdot_ua, height=0.4, alpha=0.2, color="C1")
        ax_v.plot(true_rdots[i], i, "ko", markersize=8)

    if len(resolved_rdots) > 0:
        ax_v.axvline(resolved_rdots[0], color="r", ls="--", label="Resolved range-rate")
        ax_v.legend()

    ax_v.set_yticks(y_pos)
    ax_v.set_yticklabels(prf_labels)
    ax_v.set_xlabel("Range Rate [m/s]")
    ax_v.set_title("Range-Rate Ambiguity Resolution")

    fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax_r, ax_v)
