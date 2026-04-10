"""Constant False Alarm Rate (CFAR) detection for range-Doppler maps.

Provides cell-averaging CFAR (CA-CFAR) and its variants (greatest-of and
smallest-of) for adaptive threshold detection across a 2-D RDM.  Guard and
training cells slide over the map in both range and Doppler dimensions.
"""

from __future__ import annotations
from enum import StrEnum
import numpy as np
import matplotlib.pyplot as plt


class CfarType(StrEnum):
    """Supported CFAR averaging methods."""

    CA = "CA"
    GOCA = "GOCA"
    SOCA = "SOCA"


def cfar_2d(
    rdm: np.ndarray,
    n_guard_range: int,
    n_guard_doppler: int,
    n_train_range: int,
    n_train_doppler: int,
    pfa: float,
    method: CfarType | str = CfarType.CA,
) -> tuple[np.ndarray, np.ndarray]:
    """2-D CFAR detection on a range-Doppler map.

    A sliding window estimates the local noise power from training cells
    surrounding the cell under test (CUT), with a ring of guard cells
    excluded to avoid signal leakage.  The detection threshold is set so
    that the probability of false alarm is approximately *pfa* under the
    assumption of exponential (Swerling-0) noise statistics.

    Args:
        rdm: 2-D complex or real RDM array of shape ``(n_range, n_doppler)``.
            Magnitude-squared power is computed internally if the array is
            complex.
        n_guard_range: Number of guard cells on each side of the CUT in the
            range dimension.
        n_guard_doppler: Number of guard cells on each side of the CUT in the
            Doppler dimension.
        n_train_range: Number of training cells on each side of the CUT in the
            range dimension (beyond the guard cells).
        n_train_doppler: Number of training cells on each side of the CUT in
            the Doppler dimension (beyond the guard cells).
        pfa: Desired probability of false alarm (e.g. 1e-4).
        method: CFAR variant.  One of:

            - ``"CA"`` (default): Cell-averaging — averages all training cells.
            - ``"GOCA"``: Greatest-of CA — uses the larger of the leading and
              lagging halves.  More robust at clutter edges.
            - ``"SOCA"``: Smallest-of CA — uses the smaller half.  Better
              detection in non-homogeneous environments but higher false alarm
              rate at edges.

    Returns:
        tuple: ``(detections, threshold)``:

            - **detections** (*np.ndarray*): Boolean mask of the same shape as
              *rdm*, True where the CUT exceeds the adaptive threshold.
            - **threshold** (*np.ndarray*): The computed threshold power at
              each cell, same shape as *rdm*.
    """
    method = CfarType(method)

    power = np.abs(rdm) ** 2
    n_range, n_doppler = power.shape

    # Total window half-sizes
    half_r = n_guard_range + n_train_range
    half_d = n_guard_doppler + n_train_doppler

    # Build a mask for the training cells relative to the CUT at (0, 0)
    # The outer window includes guard + training; the inner window is guard-only.
    outer_r = 2 * half_r + 1
    outer_d = 2 * half_d + 1
    mask = np.ones((outer_r, outer_d), dtype=bool)

    # Zero out the guard region (and the CUT itself)
    gr = n_guard_range
    gd = n_guard_doppler
    mask[half_r - gr : half_r + gr + 1, half_d - gd : half_d + gd + 1] = False

    n_train = int(mask.sum())

    # Threshold multiplier from desired Pfa
    # For CA-CFAR with N training cells: alpha = N * (Pfa^(-1/N) - 1)
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1)

    # Pad the power map so the window doesn't go out of bounds
    power_padded = np.pad(power, ((half_r, half_r), (half_d, half_d)), mode="wrap")

    threshold = np.zeros_like(power)

    for i in range(n_range):
        for j in range(n_doppler):
            window = power_padded[i : i + outer_r, j : j + outer_d]
            train_cells = window[mask]

            if method == CfarType.CA:
                noise_est = np.mean(train_cells)
            elif method == CfarType.GOCA:
                # Split training cells into leading (top/left) and lagging (bottom/right)
                half = len(train_cells) // 2
                noise_est = max(np.mean(train_cells[:half]), np.mean(train_cells[half:]))
            elif method == CfarType.SOCA:
                half = len(train_cells) // 2
                noise_est = min(np.mean(train_cells[:half]), np.mean(train_cells[half:]))

            threshold[i, j] = alpha * noise_est

    detections = power > threshold
    return detections, threshold


def plot_cfar(
    rdot_axis: np.ndarray,
    r_axis: np.ndarray,
    rdm: np.ndarray,
    detections: np.ndarray,
    title: str = "CFAR Detections",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot an RDM with CFAR detection markers overlaid.

    Args:
        rdot_axis: 1-D range-rate axis [m/s].
        r_axis: 1-D range axis [m].
        rdm: 2-D complex RDM (magnitude is plotted in dB).
        detections: Boolean detection mask from :func:`cfar_2d`.
        title: Plot title.

    Returns:
        The figure and axes objects.
    """
    magnitude = np.abs(rdm)
    magnitude[magnitude == 0] = np.finfo(float).tiny
    plot_data = 20 * np.log10(magnitude / magnitude.max())

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)

    mesh = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, plot_data, shading="auto")
    mesh.set_clim(-60, 0)
    cbar = fig.colorbar(mesh)
    cbar.set_label("Normalised Magnitude [dB]")

    # Overlay detection markers
    det_r, det_d = np.where(detections)
    if det_r.size > 0:
        ax.plot(
            rdot_axis[det_d] * 1e-3,
            r_axis[det_r] * 1e-3,
            "rx",
            markersize=4,
            label="CFAR detections",
        )
        ax.legend()

    ax.set_xlabel("Range Rate [km/s]")
    ax.set_ylabel("Range [km]")
    fig.tight_layout()
    return fig, ax
