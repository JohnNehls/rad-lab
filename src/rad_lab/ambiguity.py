"""Radar waveform ambiguity function computation and plotting.

The ambiguity function characterises a waveform's joint resolution in delay
(range) and Doppler frequency (velocity).  This module provides a function to
compute the narrowband ambiguity surface and helpers to visualise it.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def ambiguity_function(
    waveform: np.ndarray,
    fs: float,
    fd_max: float,
    n_fd: int = 201,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the narrowband ambiguity function of a radar waveform.

    For each Doppler shift *f_d* in a uniform grid the waveform is frequency-
    shifted and cross-correlated with the original.  The result is the squared-
    magnitude ambiguity surface normalised to a unit peak.

    Args:
        waveform: 1-D complex or real pulse samples.
        fs: Sampling rate [Hz].
        fd_max: Maximum Doppler frequency extent [Hz].  The Doppler axis spans
            ``[-fd_max, fd_max]``.
        n_fd: Number of Doppler frequency bins (should be odd for a centred
            zero-Doppler bin).  Defaults to 201.

    Returns:
        tuple: ``(tau_axis, fd_axis, ambiguity)``:

            - **tau_axis** (*np.ndarray*): 1-D delay axis [s] with length
              ``2*N - 1`` where *N* is the waveform length.
            - **fd_axis** (*np.ndarray*): 1-D Doppler frequency axis [Hz]
              with length *n_fd*.
            - **ambiguity** (*np.ndarray*): 2-D ambiguity surface
              ``(n_fd, 2*N - 1)``, peak-normalised to 1.
    """
    s = np.asarray(waveform, dtype=complex)
    N = len(s)
    n_corr = 2 * N - 1

    fd_axis = np.linspace(-fd_max, fd_max, n_fd)
    n_time = np.arange(N) / fs  # time vector for Doppler shift

    ambiguity = np.zeros((n_fd, n_corr))

    for i, fd in enumerate(fd_axis):
        # Apply Doppler shift
        s_shifted = s * np.exp(1j * 2 * np.pi * fd * n_time)
        # Cross-correlate with the reference waveform
        corr = signal.correlate(s_shifted, s, mode="full")
        ambiguity[i, :] = np.abs(corr) ** 2

    # Normalise to unit peak
    ambiguity /= ambiguity.max()

    # Delay axis centred at zero
    tau_axis = np.arange(-(N - 1), N) / fs

    return tau_axis, fd_axis, ambiguity


def plot_ambiguity(
    tau: np.ndarray,
    fd: np.ndarray,
    ambiguity: np.ndarray,
    title: str = "Ambiguity Function",
    db: bool = True,
    db_floor: float = -40,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the ambiguity surface as a colourmap.

    Args:
        tau: 1-D delay axis [s].
        fd: 1-D Doppler axis [Hz].
        ambiguity: 2-D ambiguity surface (peak-normalised).
        title: Plot title.
        db: If True, display in dB.  Defaults to True.
        db_floor: Minimum dB value for the colour scale.

    Returns:
        The figure and axes objects.
    """
    if db:
        plot_data = 10 * np.log10(np.clip(ambiguity, 10 ** (db_floor / 10), None))
        cbar_label = "Normalised Power [dB]"
        vmin, vmax = db_floor, 0
    else:
        plot_data = ambiguity
        cbar_label = "Normalised Power"
        vmin, vmax = 0, 1

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    mesh = ax.pcolormesh(tau * 1e6, fd * 1e-3, plot_data, shading="auto")
    mesh.set_clim(vmin, vmax)
    ax.set_xlabel("Delay [µs]")
    ax.set_ylabel("Doppler Frequency [kHz]")
    cbar = fig.colorbar(mesh)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig, ax


def plot_zero_cuts(
    tau: np.ndarray,
    fd: np.ndarray,
    ambiguity: np.ndarray,
    title: str = "Ambiguity Function — Zero Cuts",
    db: bool = True,
    db_floor: float = -40,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot the zero-Doppler and zero-delay cuts of the ambiguity function.

    Args:
        tau: 1-D delay axis [s].
        fd: 1-D Doppler axis [Hz].
        ambiguity: 2-D ambiguity surface (peak-normalised).
        title: Plot title.
        db: If True, display in dB.  Defaults to True.
        db_floor: Minimum dB value for the y-axis.

    Returns:
        The figure and a tuple of the two axes objects.
    """
    # Find the indices closest to zero delay and zero Doppler
    idx_fd0 = np.argmin(np.abs(fd))
    idx_tau0 = np.argmin(np.abs(tau))

    zero_doppler_cut = ambiguity[idx_fd0, :]  # autocorrelation shape
    zero_delay_cut = ambiguity[:, idx_tau0]  # Doppler response

    fig, (ax_tau, ax_fd) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    if db:
        zero_doppler_db = 10 * np.log10(np.clip(zero_doppler_cut, 10 ** (db_floor / 10), None))
        zero_delay_db = 10 * np.log10(np.clip(zero_delay_cut, 10 ** (db_floor / 10), None))
        ax_tau.plot(tau * 1e6, zero_doppler_db)
        ax_tau.set_ylabel("Normalised Power [dB]")
        ax_tau.set_ylim(db_floor, 3)
        ax_fd.plot(fd * 1e-3, zero_delay_db)
        ax_fd.set_ylabel("Normalised Power [dB]")
        ax_fd.set_ylim(db_floor, 3)
    else:
        ax_tau.plot(tau * 1e6, zero_doppler_cut)
        ax_tau.set_ylabel("Normalised Power")
        ax_fd.plot(fd * 1e-3, zero_delay_cut)
        ax_fd.set_ylabel("Normalised Power")

    ax_tau.set_xlabel("Delay [µs]")
    ax_tau.set_title("Zero-Doppler Cut (Autocorrelation)")
    ax_tau.grid(True)

    ax_fd.set_xlabel("Doppler Frequency [kHz]")
    ax_fd.set_title("Zero-Delay Cut (Doppler Response)")
    ax_fd.grid(True)

    fig.tight_layout()
    return fig, (ax_tau, ax_fd)
