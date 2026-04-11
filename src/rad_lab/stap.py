"""Space-Time Adaptive Processing (STAP) for airborne radar clutter rejection.

Simulates a multi-channel (array) pulse-Doppler radar and applies joint
spatial-temporal filtering to separate moving targets from angle-Doppler
coupled ground clutter.  Provides :func:`gen` for end-to-end simulation and
:func:`smi_weights` for the Sample Matrix Inversion adaptive processor.
"""

from __future__ import annotations

import numpy as np
from scipy import fft, signal
import matplotlib.pyplot as plt

from . import constants as c
from .rf_datacube import number_range_bins, range_axis
from .range_equation import noise_power
from .uniform_linear_arrays import steering_vector
from ._rdm_internals import _propagation_phase, create_window
from .pulse_doppler_radar import Radar
from .waveform import WaveformSample
from .returns import Target, Return
from .utilities import zero_to_smallest_float


# ---------------------------------------------------------------------------
# Datacube population
# ---------------------------------------------------------------------------


def _add_target_returns(
    datacube: np.ndarray,
    waveform: WaveformSample,
    radar: Radar,
    target: Target,
    el_pos: np.ndarray,
) -> None:
    """Inject a single target's returns into the 3-D datacube.

    For each pulse and each array element the function computes the two-way
    propagation delay (using the target's range and range-rate), applies the
    element-dependent spatial phase from the steering vector, and adds the
    scaled waveform at the appropriate range bin.

    Args:
        datacube: 3-D complex array ``(n_range, n_pulses, n_elements)``,
            modified in place.
        waveform: Waveform containing the discrete pulse samples.
        radar: Radar system parameters.
        target: Target kinematics and RCS.
        el_pos: Element positions normalised by wavelength.
    """
    n_range, n_pulses, n_elements = datacube.shape
    pulse_tx_times = np.arange(n_pulses) / radar.prf

    # Range history (constant range-rate model)
    ranges = target.range + target.range_rate * pulse_tx_times

    # Two-way delay and carrier phase per pulse
    two_way_delays = 2 * ranges / c.C
    two_way_phases = _propagation_phase(two_way_delays, radar.fcar)

    # Range bin indices (within each pulse, not absolute time)
    sample_indices = (
        np.round((two_way_delays - waveform.pulse_width / 2) * radar.sample_rate).astype(int) - 1
    )

    # Amplitude from RCS (STAP targets must have an RCS)
    amplitude = np.sqrt(target.rcs or 0.0)

    # Steering vector for this target's angle of arrival
    sv = steering_vector(el_pos, target.angle)

    n_wf = len(waveform.pulse_sample)

    for n in range(n_pulses):
        idx = sample_indices[n]
        if idx < 0 or idx + n_wf > n_range:
            continue

        # Baseband pulse with propagation phase
        pulse = amplitude * waveform.pulse_sample * np.exp(1j * two_way_phases[n])

        for e in range(n_elements):
            datacube[idx : idx + n_wf, n, e] += pulse * sv[e]


def _add_clutter(
    datacube: np.ndarray,
    radar: Radar,
    el_pos: np.ndarray,
    platform_velocity: float,
    n_clutter_patches: int = 180,
    cnr: float = 40.0,
) -> None:
    """Add angle-Doppler coupled ground clutter to the datacube.

    Models the ground as discrete isotropic scattering patches distributed
    across all range bins and azimuth angles.  Each patch has a Doppler shift
    determined by its azimuth angle relative to the platform velocity vector,
    producing the characteristic clutter ridge in angle-Doppler space.

    Clutter is injected at every range bin so that the sample covariance
    matrix estimated from training data accurately represents the clutter
    statistics — a requirement for adaptive (STAP) processing.

    Args:
        datacube: 3-D complex array ``(n_range, n_pulses, n_elements)``,
            modified in place.
        radar: Radar system parameters.
        el_pos: Element positions normalised by wavelength.
        platform_velocity: Platform ground speed [m/s].
        n_clutter_patches: Number of azimuth patches to discretise the clutter
            ring.  More patches give a smoother clutter ridge.
        cnr: Clutter-to-noise ratio per patch [linear power].  Controls the
            clutter power injected per range bin.
    """
    n_range, n_pulses, n_elements = datacube.shape

    # Clutter angles uniformly distributed in azimuth
    angles = np.linspace(-90, 90, n_clutter_patches, endpoint=False)

    wavelength = c.C / radar.fcar

    # Amplitude per patch per range bin
    patch_amplitude = np.sqrt(cnr / n_clutter_patches)

    for angle in angles:
        # Doppler shift from platform motion
        fd = 2 * platform_velocity * np.sin(np.deg2rad(angle)) / wavelength

        # Spatial steering vector for this clutter angle
        sv = steering_vector(el_pos, angle)

        # Per-pulse Doppler phase ramp
        n_vec = np.arange(n_pulses)
        doppler_phasor = np.exp(1j * 2 * np.pi * fd * n_vec / radar.prf)

        # Inject clutter at every range bin with a random phase per bin
        # (different scatterer realisations at each range)
        random_phases = np.exp(1j * 2 * np.pi * np.random.rand(n_range))

        for k in range(n_range):
            for n in range(n_pulses):
                for e in range(n_elements):
                    datacube[k, n, e] += (
                        patch_amplitude * random_phases[k] * doppler_phasor[n] * sv[e]
                    )


# ---------------------------------------------------------------------------
# STAP processing
# ---------------------------------------------------------------------------


def _space_time_snapshot(
    datacube: np.ndarray,
    range_bin: int,
) -> np.ndarray:
    """Extract space-time snapshots for a single range bin.

    The snapshot vector for range bin *k* is formed by stacking the
    pulse × element slice into a single column vector of length
    ``n_pulses * n_elements``.

    Args:
        datacube: 3-D array ``(n_range, n_pulses, n_elements)``.
        range_bin: Range bin index.

    Returns:
        1-D complex vector of length ``n_pulses * n_elements``.
    """
    # datacube[k] has shape (n_pulses, n_elements)
    return datacube[range_bin].flatten()


def _covariance_matrix(
    datacube: np.ndarray,
    range_bin: int,
    n_guard: int = 5,
) -> np.ndarray:
    """Estimate the clutter-plus-noise covariance from training range bins.

    Uses secondary data (range bins away from the cell under test) to form
    the sample covariance matrix.

    Args:
        datacube: 3-D array ``(n_range, n_pulses, n_elements)``.
        range_bin: Index of the cell under test (excluded from training).
        n_guard: Number of guard bins on each side of the CUT to exclude.

    Returns:
        Sample covariance matrix of shape ``(NM, NM)`` where
        ``NM = n_pulses * n_elements``.
    """
    n_range = datacube.shape[0]
    NM = datacube.shape[1] * datacube.shape[2]

    R = np.zeros((NM, NM), dtype=complex)
    count = 0

    for k in range(n_range):
        if abs(k - range_bin) <= n_guard:
            continue
        x = datacube[k].flatten()
        R += np.outer(x, np.conj(x))
        count += 1

    if count > 0:
        R /= count

    return R


def smi_weights(
    R: np.ndarray,
    steering_vec: np.ndarray,
    diagonal_load: float = 0.0,
) -> np.ndarray:
    """Compute adaptive STAP weights via Sample Matrix Inversion (SMI).

    The optimal weight vector maximises the output SINR for a target with
    the given space-time steering vector:

    .. math::

        \\mathbf{w} = \\mathbf{R}^{-1} \\mathbf{s}

    where **R** is the clutter-plus-noise covariance and **s** is the
    space-time steering vector.

    Args:
        R: Clutter-plus-noise covariance matrix ``(NM, NM)``.
        steering_vec: Space-time steering vector ``(NM,)`` for the desired
            target angle and Doppler.
        diagonal_load: Optional diagonal loading factor for numerical
            stability.  Added as ``diagonal_load * I`` to **R** before
            inversion.

    Returns:
        Adaptive weight vector ``(NM,)``.
    """
    NM = R.shape[0]
    R_loaded = R + diagonal_load * np.eye(NM)
    w = np.linalg.solve(R_loaded, steering_vec)
    return w


def space_time_steering_vector(
    el_pos: np.ndarray,
    n_pulses: int,
    angle: float,
    fd: float,
    prf: float,
) -> np.ndarray:
    """Build the space-time steering vector for a given angle and Doppler.

    The space-time steering vector is the Kronecker product of the temporal
    steering vector (Doppler) and the spatial steering vector (angle):

    .. math::

        \\mathbf{s} = \\mathbf{a}_t \\otimes \\mathbf{a}_s

    Args:
        el_pos: Element positions normalised by wavelength.
        n_pulses: Number of pulses in the CPI.
        angle: Target angle of arrival [degrees], 0 = broadside.
        fd: Target Doppler frequency [Hz].
        prf: Pulse repetition frequency [Hz].

    Returns:
        Space-time steering vector of length ``n_pulses * n_elements``.
    """
    # Spatial steering vector
    a_s = steering_vector(el_pos, angle)

    # Temporal steering vector
    n_vec = np.arange(n_pulses)
    a_t = np.exp(1j * 2 * np.pi * fd * n_vec / prf)

    # Kronecker product: temporal ⊗ spatial
    return np.kron(a_t, a_s)


# ---------------------------------------------------------------------------
# Conventional (non-adaptive) processing
# ---------------------------------------------------------------------------


def conventional_rdm(
    datacube: np.ndarray,
    radar: Radar,
    el_pos: np.ndarray,
    steer_angle: float = 0.0,
    window: str = "chebyshev",
    window_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a 3-D datacube with conventional beamforming + Doppler FFT.

    Applies spatial beamforming (steering vector dot product across elements)
    to collapse the array dimension, then windows and Doppler-processes the
    result to produce a standard 2-D range-Doppler map.

    Args:
        datacube: 3-D complex array ``(n_range, n_pulses, n_elements)``.
        radar: Radar system parameters.
        el_pos: Element positions normalised by wavelength.
        steer_angle: Beamforming steering angle [degrees].
        window: Doppler window function (same options as ``rdm.gen``).
        window_kwargs: Optional window parameters.

    Returns:
        tuple: ``(rdot_axis, r_axis, rdm_out)``:

            - **rdot_axis** (*np.ndarray*): Range-rate axis [m/s].
            - **r_axis** (*np.ndarray*): Range axis [m].
            - **rdm_out** (*np.ndarray*): 2-D range-Doppler map.
    """
    n_range, n_pulses, n_elements = datacube.shape

    # Beamform: dot product with steering vector across elements
    sv = steering_vector(el_pos, steer_angle)
    beamformed = np.zeros((n_range, n_pulses), dtype=complex)
    for e in range(n_elements):
        beamformed += np.conj(sv[e]) * datacube[:, :, e]

    # Doppler window
    win_mat = create_window(
        beamformed.shape, window=window, window_kwargs=window_kwargs, plot=False
    )
    beamformed *= win_mat

    # Doppler FFT
    prf = radar.sample_rate / n_range
    f_axis = fft.fftshift(fft.fftfreq(n_pulses, 1 / prf))
    beamformed[:] = fft.fftshift(fft.fft(beamformed, axis=1), axes=1)

    r_ax = range_axis(radar.sample_rate, n_range)
    rdot_axis = -c.C * f_axis / (2 * radar.fcar)

    return rdot_axis, r_ax, beamformed


# ---------------------------------------------------------------------------
# Adaptive processing
# ---------------------------------------------------------------------------


def adaptive_rdm(
    datacube: np.ndarray,
    radar: Radar,
    el_pos: np.ndarray,
    steer_angle: float = 0.0,
    n_guard: int = 5,
    diagonal_load: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a 3-D datacube with STAP (Sample Matrix Inversion).

    For each range bin and each Doppler frequency, builds a space-time
    steering vector, estimates the clutter covariance from training data,
    and applies adaptive weights to maximise the output SINR.

    Args:
        datacube: 3-D complex array ``(n_range, n_pulses, n_elements)``.
        radar: Radar system parameters.
        el_pos: Element positions normalised by wavelength.
        steer_angle: Look-direction steering angle [degrees].
        n_guard: Guard bins for covariance estimation.
        diagonal_load: Diagonal loading factor for matrix inversion.

    Returns:
        tuple: ``(rdot_axis, r_axis, rdm_out)``:

            - **rdot_axis** (*np.ndarray*): Range-rate axis [m/s].
            - **r_axis** (*np.ndarray*): Range axis [m].
            - **rdm_out** (*np.ndarray*): 2-D adaptively filtered RDM.
    """
    n_range, n_pulses, _ = datacube.shape
    prf = radar.sample_rate / n_range

    f_axis = fft.fftshift(fft.fftfreq(n_pulses, 1 / prf))
    rdot_axis = -c.C * f_axis / (2 * radar.fcar)
    r_ax = range_axis(radar.sample_rate, n_range)

    rdm_out = np.zeros((n_range, n_pulses), dtype=complex)

    for k in range(n_range):
        # Estimate covariance from training data
        R = _covariance_matrix(datacube, k, n_guard=n_guard)

        x = _space_time_snapshot(datacube, k)

        for m, fd in enumerate(f_axis):
            # Build space-time steering vector for this angle and Doppler
            s = space_time_steering_vector(el_pos, n_pulses, steer_angle, fd, prf)

            # Adaptive weights
            w = smi_weights(R, s, diagonal_load=diagonal_load)

            # Apply weights
            rdm_out[k, m] = np.dot(np.conj(w), x)

    return rdot_axis, r_ax, rdm_out


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------


def gen(
    radar: Radar,
    waveform: WaveformSample,
    return_list: list[Return],
    el_pos: np.ndarray,
    platform_velocity: float = 0.0,
    cnr: float = 0.0,
    n_clutter_patches: int = 180,
    steer_angle: float = 0.0,
    seed: int = 0,
    plot: bool = True,
    window: str = "chebyshev",
    window_kwargs: dict | None = None,
    n_guard: int = 5,
    diagonal_load: float = 1e-3,
) -> dict:
    """Generate a multi-channel RDM and process with conventional and STAP filters.

    Simulates a ULA pulse-Doppler radar with ground clutter and noise, then
    produces both a conventional beamformed RDM and an adaptively filtered
    (STAP) RDM for comparison.

    Each :class:`~rad_lab.returns.Target` in *return_list* must have an
    ``angle`` attribute giving its angle of arrival in degrees (0 = broadside).
    This can be set as ``Target(range=..., range_rate=..., rcs=..., angle=...)``.

    Args:
        radar: Radar system parameters.
        waveform: Waveform to transmit.
        return_list: List of :class:`~rad_lab.returns.Return` objects.
        el_pos: Array element positions normalised by wavelength.
        platform_velocity: Platform ground speed [m/s].  Determines the
            clutter Doppler coupling.  Set to 0 for no clutter Doppler.
        cnr: Clutter-to-noise ratio [linear power] per patch.  Set to 0
            to disable clutter.
        n_clutter_patches: Number of azimuth patches for clutter modelling.
        steer_angle: Beamforming / STAP look direction [degrees].
        seed: Random seed for reproducibility.
        plot: If True, plot conventional and adaptive RDMs side by side.
        window: Doppler window function.
        window_kwargs: Optional window parameters.
        n_guard: Guard bins for STAP covariance estimation.
        diagonal_load: Diagonal loading for STAP matrix inversion.

    Returns:
        dict with keys:

            - ``"rdot_axis"``: Range-rate axis [m/s].
            - ``"r_axis"``: Range axis [m].
            - ``"datacube"``: Raw 3-D datacube after range compression.
            - ``"conventional"``: 2-D conventional RDM.
            - ``"adaptive"``: 2-D STAP-processed RDM.
    """
    np.random.seed(seed)

    waveform.set_sample(radar.sample_rate)
    n_range = number_range_bins(radar.sample_rate, radar.prf)
    n_elements = len(el_pos)

    # Allocate 3-D datacube: range × pulses × elements
    datacube = np.zeros((n_range, radar.n_pulses, n_elements), dtype=np.complex64)

    # Inject target returns
    for ret in return_list:
        _add_target_returns(datacube, waveform, radar, ret.target, el_pos)

    # Inject clutter
    if cnr > 0:
        _add_clutter(
            datacube,
            radar,
            el_pos,
            platform_velocity,
            n_clutter_patches,
            cnr,
        )

    # Add thermal noise
    noise_pwr = noise_power(waveform.bw, radar.noise_factor, radar.op_temp)
    noise_volt = np.sqrt(c.RADAR_LOAD * noise_pwr)
    datacube += np.random.uniform(low=-1, high=1, size=datacube.shape) * noise_volt

    # Range compression (per element)
    kernel = np.conj(waveform.pulse_sample)[::-1]
    for e in range(n_elements):
        dc_2d = datacube[:, :, e]
        dc_2d[:] = signal.fftconvolve(dc_2d, kernel.reshape(-1, 1), mode="same", axes=0)

    # Conventional processing
    rdot_axis, r_axis, conv_rdm = conventional_rdm(
        datacube.copy(),
        radar,
        el_pos,
        steer_angle,
        window,
        window_kwargs,
    )

    # Adaptive (STAP) processing
    _, _, adapt_rdm = adaptive_rdm(
        datacube.copy(),
        radar,
        el_pos,
        steer_angle,
        n_guard,
        diagonal_load,
    )

    if plot:
        plot_comparison(rdot_axis, r_axis, conv_rdm, adapt_rdm)

    return {
        "rdot_axis": rdot_axis,
        "r_axis": r_axis,
        "datacube": datacube,
        "conventional": conv_rdm,
        "adaptive": adapt_rdm,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(
    rdot_axis: np.ndarray,
    r_axis: np.ndarray,
    conv_rdm: np.ndarray,
    adapt_rdm: np.ndarray,
    cbar_min: float = -60,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot conventional and adaptive RDMs side by side.

    Args:
        rdot_axis: Range-rate axis [m/s].
        r_axis: Range axis [m].
        conv_rdm: 2-D conventional RDM.
        adapt_rdm: 2-D STAP-processed RDM.
        cbar_min: Minimum colorbar value [dB].

    Returns:
        The figure and a tuple of the two axes.
    """
    fig, (ax_conv, ax_stap) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Conventional vs. STAP Processing")

    for ax, data, title in [
        (ax_conv, conv_rdm, "Conventional (Beamform + FFT)"),
        (ax_stap, adapt_rdm, "STAP (SMI)"),
    ]:
        magnitude = np.abs(data)
        zero_to_smallest_float(magnitude)
        plot_data = 20 * np.log10(magnitude / magnitude.max())

        mesh = ax.pcolormesh(rdot_axis * 1e-3, r_axis * 1e-3, plot_data, shading="auto")
        mesh.set_clim(cbar_min, 0)
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Normalised Magnitude [dB]")
        ax.set_title(title)
        ax.set_xlabel("Range Rate [km/s]")
        ax.set_ylabel("Range [km]")

    fig.tight_layout()
    return fig, (ax_conv, ax_stap)
