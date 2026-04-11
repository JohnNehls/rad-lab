"""Moving Target Indication (MTI) cancellers.

Provides FIR highpass filters applied along the slow-time (pulse) dimension
of a radar datacube to suppress stationary clutter while preserving moving
targets.

An *N*-pulse canceller is a binomial FIR filter of order *N − 1* whose
coefficients are the alternating-sign binomial coefficients:

.. math::

    h[k] = (-1)^k \\binom{N-1}{k}, \\quad k = 0, \\ldots, N-1

==========  ======================  ============================
Canceller   Weights                 Null depth at DC
==========  ======================  ============================
2-pulse     ``[1, −1]``             Single null (1st-order)
3-pulse     ``[1, −2, 1]``          Double null (2nd-order)
4-pulse     ``[1, −3, 3, −1]``      Triple null (3rd-order)
==========  ======================  ============================

Higher-order cancellers suppress broader clutter spectra but also widen
the blind-speed notches around multiples of the PRF, reducing the
Doppler region where targets can be detected.

Reference: Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
2014, §7.3 — "MTI with Delay Line Cancellers".
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


# ---------------------------------------------------------------------------
# Canceller weights
# ---------------------------------------------------------------------------


def canceller_weights(n_pulses):
    """Binomial MTI canceller weights for an *n_pulses*-pulse canceller.

    Args:
        n_pulses: Number of pulses consumed per output sample (filter
            length).  Must be >= 2.

    Returns:
        1-D array of integer weights, length *n_pulses*.
    """
    if n_pulses < 2:
        raise ValueError("n_pulses must be >= 2 for an MTI canceller.")
    order = n_pulses - 1
    return np.array([(-1) ** k * comb(order, k, exact=True) for k in range(n_pulses)])


# ---------------------------------------------------------------------------
# Apply canceller to datacube
# ---------------------------------------------------------------------------


def apply(datacube, weights):
    """Apply an MTI canceller to a datacube along the slow-time axis.

    Convolves each range bin's slow-time sequence with *weights* using
    ``"valid"`` mode, so the output has fewer pulses than the input
    (reduced by ``len(weights) − 1``).

    Args:
        datacube: 2-D complex array of shape ``(n_range_bins, n_pulses)``.
        weights: 1-D array of canceller weights (e.g. from
            :func:`canceller_weights`).

    Returns:
        Filtered datacube of shape
        ``(n_range_bins, n_pulses − len(weights) + 1)``.
    """
    weights = np.asarray(weights, dtype=complex)
    n_range_bins, n_pulses = datacube.shape
    n_out = n_pulses - len(weights) + 1
    if n_out < 1:
        raise ValueError(
            f"Canceller length ({len(weights)}) exceeds number of pulses ({n_pulses})."
        )

    out = np.empty((n_range_bins, n_out), dtype=datacube.dtype)
    for i in range(n_range_bins):
        out[i, :] = np.convolve(datacube[i, :], weights, mode="valid")
    return out


# ---------------------------------------------------------------------------
# Frequency response
# ---------------------------------------------------------------------------


def frequency_response(weights, n_freq=1024, prf=1.0):
    """Compute the frequency response of an MTI canceller.

    Args:
        weights: 1-D array of canceller weights.
        n_freq: Number of frequency points.
        prf: Pulse repetition frequency [Hz].  The response is evaluated
            over :math:`[-\\text{PRF}/2,\\; \\text{PRF}/2)`.

    Returns:
        (f_axis, H): Frequency axis [Hz] and complex frequency response.
    """
    weights = np.asarray(weights, dtype=complex)
    n = len(weights)
    f_axis = np.linspace(-prf / 2, prf / 2, n_freq, endpoint=False)
    # H(f) = sum_k w[k] * exp(-j*2*pi*f*k/PRF)
    k = np.arange(n)
    H = np.array([np.sum(weights * np.exp(-1j * 2 * np.pi * f * k / prf)) for f in f_axis])
    return f_axis, H


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_frequency_response(
    weights_list,
    labels=None,
    prf=1.0,
    n_freq=1024,
    title="MTI Canceller Frequency Response",
):
    """Plot the magnitude response of one or more MTI cancellers.

    Args:
        weights_list: List of weight arrays to compare.
        labels: List of labels for the legend.  Defaults to
            ``"N-pulse"`` based on filter length.
        prf: Pulse repetition frequency [Hz].
        n_freq: Number of frequency points.
        title: Figure title.

    Returns:
        (fig, ax) tuple.
    """
    if labels is None:
        labels = [f"{len(w)}-pulse" for w in weights_list]

    fig, ax = plt.subplots(figsize=(8, 5))

    for weights, label in zip(weights_list, labels):
        f_axis, H = frequency_response(weights, n_freq, prf)
        mag = np.abs(H)
        mag[mag == 0] = np.finfo(float).tiny
        ax.plot(f_axis / prf, 20 * np.log10(mag / mag.max()), label=label)

    ax.set_xlabel("Normalised Doppler frequency ($f / f_{PRF}$)")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title(title)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-60, 3)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax
