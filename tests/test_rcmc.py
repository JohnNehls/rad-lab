"""Validate Range Cell Migration Correction against the closed-form formula.

A perfectly range-compressed point-target signal is synthesised directly:
at each pulse ``eta`` the energy is a unit sinc lobe centred on the exact
slant range ``R(eta) = sqrt(R0^2 + v^2 eta^2)`` modulated by the two-way
carrier phase.  Without RCMC the peak range traces a parabola in the
range-Doppler domain; after RCMC it must collapse to a flat line at R0.
"""

import numpy as np
from scipy import fft

from rad_lab import SarRadar, constants as c
from rad_lab._sar_internals import rcmc


def _synthesise_range_compressed_target(sar_radar: SarRadar, R0: float):
    """Return ``(datacube, range_axis, dR_grid)`` for a single point target."""
    n_pulses = sar_radar.n_pulses
    n_range_bins = int(sar_radar.sample_rate / sar_radar.prf)
    dR_grid = c.C / (2 * sar_radar.sample_rate)
    range_axis = np.arange(1, n_range_bins + 1) * dR_grid
    v = sar_radar.platform_velocity
    lam = sar_radar.wavelength

    eta = (np.arange(n_pulses) - n_pulses // 2) / sar_radar.prf
    R_eta = np.sqrt(R0**2 + (v * eta) ** 2)

    sinc_env = np.sinc((range_axis[:, None] - R_eta[None, :]) / dR_grid)
    carrier = np.exp(-1j * 4 * np.pi / lam * R_eta)
    datacube = (sinc_env * carrier[None, :]).astype(np.complex64)

    return datacube, range_axis, dR_grid


def test_rcmc_straightens_migration_trajectory():
    """Peak range vs Doppler: parabolic before RCMC, flat at R0 after."""
    # Migration is independent of platform velocity, but PRF must scale
    # with v to avoid Doppler aliasing — and n_range_bins = sample_rate/PRF
    # therefore shrinks with higher v.  Pushing v to ~Mach 12 (unphysical
    # but synthetic) cuts the datacube to ~7 MB while keeping ~1.5 cells
    # of migration.
    sar_radar = SarRadar(
        fcar=10e9,
        tx_power=1,
        tx_gain=1,
        rx_gain=1,
        op_temp=290,
        sample_rate=20e6,
        noise_factor=1,
        total_losses=1,
        prf=200000,
        platform_velocity=4000,
        aperture_length=180,
        platform_altitude=1,
    )
    R0 = 360.0

    datacube, range_axis, dR_grid = _synthesise_range_compressed_target(sar_radar, R0)
    rdm_before = fft.fft(datacube.copy(), axis=1)
    rcmc(datacube, sar_radar, range_axis)
    rdm_after = fft.fft(datacube, axis=1)

    f_eta = fft.fftfreq(datacube.shape[1], d=1.0 / sar_radar.prf)

    # Restrict to ~80% of the target's Doppler bandwidth so that argmax sees
    # the trajectory rather than spectral leakage at the band edges.
    f_eta_band = (
        2
        * sar_radar.platform_velocity
        * sar_radar.aperture_length
        / (R0 * sar_radar.wavelength)
        / 2
        * 0.8
    )
    in_band = np.abs(f_eta) < f_eta_band

    k_target = int(np.round(R0 / dR_grid - 1))
    search = slice(max(0, k_target - 3), k_target + 8)
    peaks_before = np.argmax(np.abs(rdm_before[search, :]), axis=0) + search.start
    peaks_after = np.argmax(np.abs(rdm_after[search, :]), axis=0) + search.start

    span_before = peaks_before[in_band].max() - peaks_before[in_band].min()
    span_after = peaks_after[in_band].max() - peaks_after[in_band].min()

    # Synthetic config has ~1.7 cells of migration → before-trajectory spans ≥ 1 cell
    assert span_before >= 1, f"Synthetic should show migration; got span {span_before} cells"
    # RCMC must collapse the trajectory to within ±1 cell across the target band
    assert span_after <= 1, (
        f"After RCMC, peak range should sit within 1 cell of R0 across the "
        f"target Doppler band (span {span_after} cells)"
    )
    # And the after-span must be smaller than the before-span
    assert span_after < span_before


def test_rcmc_is_identity_at_zero_doppler():
    """At f_eta = 0 the migration is zero, so the RCMC shift should be too."""
    sar_radar = SarRadar(
        fcar=10e9,
        tx_power=1,
        tx_gain=1,
        rx_gain=1,
        op_temp=290,
        sample_rate=20e6,
        noise_factor=1,
        total_losses=1,
        prf=200000,
        platform_velocity=4000,
        aperture_length=180,
        platform_altitude=1,
    )
    R0 = 360.0

    datacube, range_axis, _ = _synthesise_range_compressed_target(sar_radar, R0)
    rdm_before = fft.fft(datacube.copy(), axis=1)
    rcmc(datacube, sar_radar, range_axis)
    rdm_after = fft.fft(datacube, axis=1)

    # f_eta=0 lives at index 0 in unshifted FFT order
    np.testing.assert_allclose(rdm_after[:, 0], rdm_before[:, 0], rtol=1e-3, atol=1e-3)


def test_rcmc_produces_finite_values():
    """RCMC must not introduce NaN or Inf, even on noise-only input."""
    sar_radar = SarRadar(
        fcar=10e9,
        tx_power=1,
        tx_gain=1,
        rx_gain=1,
        op_temp=290,
        sample_rate=20e6,
        noise_factor=1,
        total_losses=1,
        prf=200000,
        platform_velocity=4000,
        aperture_length=180,
        platform_altitude=1,
    )
    n_range_bins = int(sar_radar.sample_rate / sar_radar.prf)
    range_axis = np.arange(1, n_range_bins + 1) * (c.C / (2 * sar_radar.sample_rate))

    rng = np.random.default_rng(0)
    n_pulses = sar_radar.n_pulses
    datacube = (
        rng.standard_normal((n_range_bins, n_pulses))
        + 1j * rng.standard_normal((n_range_bins, n_pulses))
    ).astype(np.complex64)

    rcmc(datacube, sar_radar, range_axis)
    assert np.all(np.isfinite(datacube))
