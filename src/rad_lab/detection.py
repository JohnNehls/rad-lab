"""Detection theory for radar signal processing.

Implements detection probability models for non-fluctuating and fluctuating
targets (Swerling 0, I, III), Albersheim's closed-form SNR approximation,
and plotting utilities for ROC curves and Pd-vs-SNR curves.

All models assume a square-law detector operating on the output of a
coherent processor (matched filter + Doppler FFT).  The input SNR is the
**post-integration** signal-to-noise ratio — for coherent integration of
*N* pulses, this is *N* times the single-pulse SNR.

Swerling model summary
----------------------
==========  =============================  =============================
Model       RCS fluctuation                Decorrelation
==========  =============================  =============================
0 (or V)    Non-fluctuating (constant)     —
I           Chi-squared, 2 DOF (Rayleigh)  Scan-to-scan (slow)
II          Chi-squared, 2 DOF (Rayleigh)  Pulse-to-pulse (fast)
III         Chi-squared, 4 DOF (Rician)    Scan-to-scan (slow)
IV          Chi-squared, 4 DOF (Rician)    Pulse-to-pulse (fast)
==========  =============================  =============================

References
----------
Richards, M. A., *Fundamentals of Radar Signal Processing*, 2nd ed.,
McGraw-Hill, 2014, Ch. 5–6.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, ncx2
from scipy.special import gammaincc, gammaln
from scipy.optimize import brentq
from scipy import integrate


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


def threshold_factor(pfa):
    r"""Normalised detection threshold for a square-law detector.

    For a single sample of complex Gaussian noise passed through a
    square-law detector, the false-alarm probability is

    .. math:: P_{fa} = e^{-V_T}

    so the normalised threshold is :math:`V_T = -\ln P_{fa}`.

    Args:
        pfa: Probability of false alarm (scalar or array).

    Returns:
        Normalised threshold :math:`V_T` (same shape as *pfa*).
    """
    return -np.log(pfa)


def threshold_factor_nci(pfa, n_pulses):
    r"""Normalised detection threshold after non-coherent integration.

    After summing the square-law outputs of *n_pulses* independent looks,
    the noise-only test statistic follows a chi-squared distribution with
    :math:`2 N` degrees of freedom.  The threshold is the inverse survival
    function at the desired :math:`P_{fa}`.

    Args:
        pfa: Probability of false alarm.
        n_pulses: Number of non-coherently integrated pulses.

    Returns:
        Normalised threshold (divide by 2 to get the per-sample scale).
    """
    return chi2.isf(pfa, 2 * n_pulses) / 2


# ---------------------------------------------------------------------------
# Detection probability — single-look (post coherent integration)
# ---------------------------------------------------------------------------


def pd_swerling0(snr, pfa):
    r"""Detection probability for a non-fluctuating target (Swerling 0 / V).

    Uses the non-central chi-squared distribution:

    .. math::

        P_d = 1 - F_{\chi^2_{nc}}(2 V_T;\; \text{df}=2,\; \lambda=2\,\text{SNR})

    where :math:`V_T = -\ln P_{fa}`.

    Args:
        snr: Post-integration SNR (linear, not dB).  Scalar or array.
        pfa: Probability of false alarm (scalar).

    Returns:
        Detection probability (same shape as *snr*).
    """
    snr = np.asarray(snr, dtype=float)
    vt = threshold_factor(pfa)
    # Guard against SNR exactly 0 — ncx2 needs nc > 0
    nc = np.maximum(2 * snr, 1e-30)
    return ncx2.sf(2 * vt, df=2, nc=nc)


def pd_swerling1(snr, pfa):
    r"""Detection probability for a Swerling I target.

    The RCS follows an exponential (chi-squared 2 DOF) distribution,
    decorrelating scan-to-scan.  Closed-form result:

    .. math:: P_d = P_{fa}^{1/(1 + \text{SNR})}

    Args:
        snr: Average post-integration SNR (linear).
        pfa: Probability of false alarm.

    Returns:
        Detection probability.
    """
    snr = np.asarray(snr, dtype=float)
    return pfa ** (1 / (1 + snr))


def pd_swerling3(snr, pfa):
    r"""Detection probability for a Swerling III target.

    The RCS follows a chi-squared distribution with 4 DOF (two dominant
    scatterers), decorrelating scan-to-scan.  The closed-form result is
    obtained by averaging the Swerling 0 detection probability over the
    :math:`\Gamma(2,\, \text{SNR}/2)` RCS distribution:

    .. math::

        P_d = e^{-2 V_T / (S+2)}
              \left(1 + \frac{2\,S\,V_T}{(S+2)^2}\right)

    where :math:`V_T = -\ln P_{fa}` and :math:`S = \text{SNR}`.

    Args:
        snr: Average post-integration SNR (linear).
        pfa: Probability of false alarm.

    Returns:
        Detection probability.
    """
    snr = np.asarray(snr, dtype=float)
    vt = threshold_factor(pfa)
    s2 = snr + 2
    return np.exp(-2 * vt / s2) * (1 + 2 * snr * vt / s2**2)


# ---------------------------------------------------------------------------
# Detection probability — non-coherent integration (Swerling 0)
# ---------------------------------------------------------------------------


def pd_swerling0_nci(snr_per_pulse, pfa, n_pulses):
    r"""Detection probability for Swerling 0 with non-coherent integration.

    After summing the square-law outputs of *n_pulses* independent pulses,
    the test statistic under :math:`H_1` follows a non-central chi-squared
    distribution with :math:`2 N` DOF and non-centrality
    :math:`\lambda = 2 N \,\text{SNR}_1`.

    Args:
        snr_per_pulse: Single-pulse SNR (linear).
        pfa: Probability of false alarm.
        n_pulses: Number of non-coherently integrated pulses.

    Returns:
        Detection probability.
    """
    snr_per_pulse = np.asarray(snr_per_pulse, dtype=float)
    thresh = chi2.isf(pfa, 2 * n_pulses)
    nc = np.maximum(2 * n_pulses * snr_per_pulse, 1e-30)
    return ncx2.sf(thresh, 2 * n_pulses, nc)


def pd_swerling1_nci(snr_per_pulse, pfa, n_pulses):
    r"""Detection probability for Swerling I with non-coherent integration.

    Swerling I: RCS is constant within a dwell (scan-to-scan fluctuation)
    with chi-squared 2 DOF (exponential) distribution.  All *N* pulses
    see the same random RCS, so the conditional :math:`P_d` is the
    Swerling 0 NCI result averaged over the exponential RCS:

    .. math::

        P_d = \int_0^\infty P_{d,\text{Sw0}}(\sigma,\, P_{fa},\, N)
              \;\frac{1}{\overline{\text{SNR}}}
              e^{-\sigma / \overline{\text{SNR}}} \, d\sigma

    Args:
        snr_per_pulse: Average single-pulse SNR (linear).
        pfa: Probability of false alarm.
        n_pulses: Number of non-coherently integrated pulses.

    Returns:
        Detection probability.
    """
    snr_per_pulse = np.asarray(snr_per_pulse, dtype=float)
    scalar = snr_per_pulse.ndim == 0
    snr_arr = np.atleast_1d(snr_per_pulse)

    result = np.empty_like(snr_arr)
    for i, snr_avg in enumerate(snr_arr):
        if snr_avg < 1e-10:
            result[i] = pfa
            continue

        # Substitution t = σ / SNR_avg normalises the weight to Exp(1)
        def integrand(t, _snr=snr_avg):
            return pd_swerling0_nci(t * _snr, pfa, n_pulses) * np.exp(-t)

        result[i], _ = integrate.quad(integrand, 0, np.inf, limit=100)

    return float(result[0]) if scalar else result


def pd_swerling2(snr_per_pulse, pfa, n_pulses):
    r"""Detection probability for Swerling II with non-coherent integration.

    Swerling II: RCS fluctuates pulse-to-pulse with chi-squared 2 DOF
    (exponential) distribution.  Each pulse sees an independent RCS, so
    after square-law detection each output is independently
    :math:`\text{Exp}(1 + \overline{\text{SNR}})`.  The NCI sum follows
    a Gamma distribution and the detection probability is:

    .. math::

        P_d = Q\!\bigl(N,\; V_T / (1 + \overline{\text{SNR}})\bigr)

    where :math:`Q(a, x) = \Gamma(a, x) / \Gamma(a)` is the regularised
    upper incomplete gamma function.

    For :math:`N = 1` this reduces to the Swerling I single-look result.

    Args:
        snr_per_pulse: Average single-pulse SNR (linear).
        pfa: Probability of false alarm.
        n_pulses: Number of non-coherently integrated pulses.

    Returns:
        Detection probability.
    """
    snr_per_pulse = np.asarray(snr_per_pulse, dtype=float)
    vt = threshold_factor_nci(pfa, n_pulses)
    return gammaincc(n_pulses, np.maximum(vt / (1 + snr_per_pulse), 0))


def pd_swerling3_nci(snr_per_pulse, pfa, n_pulses):
    r"""Detection probability for Swerling III with non-coherent integration.

    Swerling III: RCS is constant within a dwell (scan-to-scan) with
    chi-squared 4 DOF distribution.  Conditional :math:`P_d` is the
    Swerling 0 NCI result averaged over the :math:`\Gamma(2,\,
    \overline{\text{SNR}}/2)` RCS distribution:

    .. math::

        P_d = \int_0^\infty P_{d,\text{Sw0}}(\sigma,\, P_{fa},\, N)
              \;\frac{4\,\sigma}{\overline{\text{SNR}}^2}
              \,e^{-2\sigma / \overline{\text{SNR}}} \, d\sigma

    Args:
        snr_per_pulse: Average single-pulse SNR (linear).
        pfa: Probability of false alarm.
        n_pulses: Number of non-coherently integrated pulses.

    Returns:
        Detection probability.
    """
    snr_per_pulse = np.asarray(snr_per_pulse, dtype=float)
    scalar = snr_per_pulse.ndim == 0
    snr_arr = np.atleast_1d(snr_per_pulse)

    result = np.empty_like(snr_arr)
    for i, snr_avg in enumerate(snr_arr):
        if snr_avg < 1e-10:
            result[i] = pfa
            continue
        beta = snr_avg / 2

        # Substitution t = σ / β normalises the weight to Gamma(2, 1)
        def integrand(t, _beta=beta):
            return pd_swerling0_nci(t * _beta, pfa, n_pulses) * t * np.exp(-t)

        result[i], _ = integrate.quad(integrand, 0, np.inf, limit=100)

    return float(result[0]) if scalar else result


def pd_swerling4(snr_per_pulse, pfa, n_pulses):
    r"""Detection probability for Swerling IV with non-coherent integration.

    Swerling IV: RCS fluctuates pulse-to-pulse with chi-squared 4 DOF
    distribution.  Each of the *N* pulses sees an independent RCS draw
    :math:`\sigma_k \sim \Gamma(2,\, \overline{\text{SNR}}/2)`.  The NCI
    sum of square-law outputs is non-central chi-squared with the total
    non-centrality equal to :math:`2 \sum \sigma_k`.  Since the sum
    :math:`S = \sum \sigma_k \sim \Gamma(2N,\, \overline{\text{SNR}}/2)`,
    the detection probability is:

    .. math::

        P_d = \int_0^\infty P\bigl(\chi^2_{2N,\,2s} > T\bigr)
              \; f_{\Gamma(2N,\,\beta)}(s) \, ds

    For :math:`N = 1` this reduces to the Swerling III single-look result.

    Args:
        snr_per_pulse: Average single-pulse SNR (linear).
        pfa: Probability of false alarm.
        n_pulses: Number of non-coherently integrated pulses.

    Returns:
        Detection probability.
    """
    snr_per_pulse = np.asarray(snr_per_pulse, dtype=float)
    scalar = snr_per_pulse.ndim == 0
    snr_arr = np.atleast_1d(snr_per_pulse)

    thresh = chi2.isf(pfa, 2 * n_pulses)
    df = 2 * n_pulses

    result = np.empty_like(snr_arr)
    for i, snr_avg in enumerate(snr_arr):
        if snr_avg < 1e-10:
            result[i] = pfa
            continue
        beta = snr_avg / 2
        shape = 2 * n_pulses  # Gamma shape for sum of N iid Gamma(2, β)

        # Substitution t = s / β normalises the weight to Gamma(shape, 1)
        def integrand(t, _shape=shape, _beta=beta):
            nc = max(2 * t * _beta, 1e-30)
            pd_cond = ncx2.sf(thresh, df, nc)
            log_w = (_shape - 1) * np.log(max(t, 1e-300)) - t - gammaln(_shape)
            return pd_cond * np.exp(log_w)

        result[i], _ = integrate.quad(integrand, 0, np.inf, limit=200)

    return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# Required SNR (exact numerical inverse)
# ---------------------------------------------------------------------------


def required_snr(pd, pfa, model="swerling0"):
    r"""Required post-integration SNR for a given :math:`P_d` and :math:`P_{fa}`.

    Numerically inverts the exact detection probability function to find
    the SNR that achieves the target :math:`P_d`.

    Args:
        pd: Desired detection probability (0 < pd < 1).
        pfa: Desired false-alarm probability (0 < pfa < 1).
        model: Swerling model — ``"swerling0"``, ``"swerling1"``, or
            ``"swerling3"``.  Defaults to ``"swerling0"``.

    Returns:
        Required post-integration SNR in **dB**.
    """
    dispatch = {
        "swerling0": pd_swerling0,
        "swerling1": pd_swerling1,
        "swerling3": pd_swerling3,
    }
    func = dispatch[model]

    def objective(snr_db):
        snr_lin = 10 ** (snr_db / 10)
        return func(snr_lin, pfa) - pd

    return float(brentq(objective, -20, 60))


def required_snr_nci(pd, pfa, n_pulses, model="swerling0"):
    r"""Required single-pulse SNR for non-coherent integration.

    Numerically inverts the NCI detection probability function for the
    specified Swerling model to find the per-pulse SNR that achieves
    the target :math:`P_d`.

    Args:
        pd: Desired detection probability.
        pfa: Desired false-alarm probability.
        n_pulses: Number of non-coherently integrated pulses.
        model: Swerling model — ``"swerling0"``, ``"swerling1"``,
            ``"swerling2"``, ``"swerling3"``, or ``"swerling4"``.

    Returns:
        Required single-pulse SNR in **dB**.
    """
    dispatch = {
        "swerling0": lambda s: pd_swerling0_nci(s, pfa, n_pulses),
        "swerling1": lambda s: pd_swerling1_nci(s, pfa, n_pulses),
        "swerling2": lambda s: pd_swerling2(s, pfa, n_pulses),
        "swerling3": lambda s: pd_swerling3_nci(s, pfa, n_pulses),
        "swerling4": lambda s: pd_swerling4(s, pfa, n_pulses),
    }
    func = dispatch[model]

    def objective(snr_db):
        snr_lin = 10 ** (snr_db / 10)
        return func(snr_lin) - pd

    return float(brentq(objective, -20, 60))


# ---------------------------------------------------------------------------
# Albersheim's equation (closed-form approximation)
# ---------------------------------------------------------------------------


def albersheim(pd, pfa, n_pulses=1):
    r"""Approximate single-pulse SNR via Albersheim's equation.

    Albersheim's equation (Richards, Eq. 5.28):

    .. math::

        \text{SNR}_1\,[\text{dB}] = -5.2 +
            \left(6.2 + \frac{4.54}{\sqrt{N}}\right)
            \log_{10}\!\bigl(A + 0.12\,A\,B + 1.7\,B\bigr)

    where :math:`A = \ln(0.62 / P_{fa})` and
    :math:`B = \ln\bigl(P_d / (1 - P_d)\bigr)`.

    .. note::

        This is a *closed-form approximation* for non-coherent integration.
        It is most accurate for moderate :math:`N` (roughly 10–1000) and can
        deviate by several dB for :math:`N = 1`.  Use :func:`required_snr`
        or :func:`required_snr_nci` for exact results.

    Args:
        pd: Desired detection probability.
        pfa: Desired false-alarm probability.
        n_pulses: Number of non-coherently integrated pulses.

    Returns:
        Approximate single-pulse SNR in **dB**.
    """
    a = np.log(0.62 / pfa)
    b = np.log(pd / (1 - pd))
    return -5.2 + (6.2 + 4.54 / np.sqrt(n_pulses)) * np.log10(a + 0.12 * a * b + 1.7 * b)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_pd_vs_snr(
    snr_db,
    pfa,
    models=None,
    n_nci=None,
    title=None,
):
    r"""Plot detection probability vs SNR for one or more Swerling models.

    Args:
        snr_db: 1-D array of SNR values in dB.
        pfa: Probability of false alarm.
        models: List of model names to plot.  Accepted values:
            ``"swerling0"``, ``"swerling1"``, ``"swerling3"``.
            Defaults to all three.
        n_nci: If given, also plot Swerling 0 with *n_nci* non-coherently
            integrated pulses (SNR axis is per-pulse in this case).
        title: Figure title.  Defaults to
            ``"Pd vs SNR (Pfa = <pfa>)"``.

    Returns:
        (fig, ax) tuple.
    """
    if models is None:
        models = ["swerling0", "swerling1", "swerling3"]
    if title is None:
        title = f"$P_d$ vs SNR ($P_{{fa}} = {pfa:.0e}$)"

    snr_db = np.asarray(snr_db, dtype=float)
    snr_lin = 10 ** (snr_db / 10)

    dispatch = {
        "swerling0": ("Swerling 0", pd_swerling0),
        "swerling1": ("Swerling I", pd_swerling1),
        "swerling3": ("Swerling III", pd_swerling3),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for key in models:
        label, func = dispatch[key]
        ax.plot(snr_db, func(snr_lin, pfa), label=label)

    if n_nci is not None:
        pd_nci = pd_swerling0_nci(snr_lin, pfa, n_nci)
        ax.plot(snr_db, pd_nci, "--", label=f"Swerling 0, NCI N={n_nci}")

    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("$P_d$")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_roc(
    snr_db_values,
    pfa_range=None,
    model="swerling0",
    title=None,
):
    r"""Plot receiver operating characteristic (ROC) curves.

    Each curve shows :math:`P_d` vs :math:`P_{fa}` for a fixed SNR.

    Args:
        snr_db_values: Sequence of SNR values in dB, one curve each.
        pfa_range: 1-D array of :math:`P_{fa}` values.  Defaults to
            ``np.logspace(-8, -1, 200)``.
        model: Swerling model — ``"swerling0"``, ``"swerling1"``, or
            ``"swerling3"``.
        title: Figure title.

    Returns:
        (fig, ax) tuple.
    """
    if pfa_range is None:
        pfa_range = np.logspace(-8, -1, 200)
    if title is None:
        model_labels = {
            "swerling0": "Swerling 0",
            "swerling1": "Swerling I",
            "swerling3": "Swerling III",
        }
        title = f"ROC Curves — {model_labels.get(model, model)}"

    dispatch = {
        "swerling0": pd_swerling0,
        "swerling1": pd_swerling1,
        "swerling3": pd_swerling3,
    }
    func = dispatch[model]

    fig, ax = plt.subplots(figsize=(8, 5))

    for snr_db in snr_db_values:
        snr_lin = 10 ** (snr_db / 10)
        pd_vals = np.array([func(snr_lin, pfa) for pfa in pfa_range])
        ax.semilogx(pfa_range, pd_vals, label=f"SNR = {snr_db:.0f} dB")

    ax.set_xlabel("$P_{fa}$")
    ax.set_ylabel("$P_d$")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, ax


def plot_required_snr_vs_n(
    pd,
    pfa,
    n_pulses_range=None,
    title=None,
):
    r"""Plot required single-pulse SNR vs number of non-coherently integrated pulses.

    Uses the exact numerical inverse (:func:`required_snr_nci`) and
    overlays Albersheim's closed-form approximation for comparison.

    Args:
        pd: Desired detection probability.
        pfa: Desired false-alarm probability.
        n_pulses_range: 1-D array of pulse counts.  Defaults to
            ``np.arange(1, 65)``.
        title: Figure title.

    Returns:
        (fig, ax) tuple.
    """
    if n_pulses_range is None:
        n_pulses_range = np.arange(1, 65)
    if title is None:
        title = f"Required SNR$_1$ vs $N$ ($P_d = {pd}$, $P_{{fa}} = {pfa:.0e}$)"

    snr_exact = [required_snr_nci(pd, pfa, n) for n in n_pulses_range]
    snr_approx = [albersheim(pd, pfa, n) for n in n_pulses_range]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_pulses_range, snr_exact, label="Exact (numerical)")
    ax.plot(n_pulses_range, snr_approx, "--", label="Albersheim (approx.)")
    ax.set_xlabel("Number of non-coherently integrated pulses $N$")
    ax.set_ylabel("Required single-pulse SNR$_1$ [dB]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax
