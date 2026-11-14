#!/usr/bin/env python
"""Range profiles and non-coherent integration (NCI).

Build range profiles from the range-time matrix (`rdm.gen_rtm`, no Doppler
FFT) by summing pulse power, and compare one pulse with N.  Then check the
simulated false-alarm and detection rates against `detection.py` theory.

Notes:
  - NCI sums |x|^2 over pulses, so phase (and therefore Doppler) does not
    matter.  That is why search radars use it: the target's speed is not
    known while scanning.
  - With per-pulse SNR normalisation (`rdm.rtm_to_snr`) the noise-only NCI
    output has mean N, and a target bin has mean N * (1 + SNR_1).  The
    threshold `threshold_factor_nci(pfa, N)` applies to the summed profile
    directly.
  - The NCI gain is less than the factor of N that coherent integration
    gives, and the gap (the NCI loss) grows with N; compare the table at the
    end with exercise 6_1.
  - Targets sit on range-bin centres and are spaced by one pulse length (20
    samples), so their matched-filter outputs do not overlap.  The simulator
    rounds return delays to the nearest sample, so there is no straddle loss
    in any case.
  - The matched filter correlates neighbouring range bins, so the noise-only
    samples are far fewer independent looks than their count suggests; the
    measured noise mean and Pfa scatter by a few percent around theory.
  - With ``window="none"`` the same profile is the RDM power summed over
    Doppler divided by N (Parseval); see tests/test_range_profile.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm, detection, Radar, Target, Return, lfm_waveform
from rad_lab import constants as c
from rad_lab.range_equation import snr_range_eqn

bw = 10e6  # waveform bandwidth [Hz]
T = 1.0e-6  # pulse width [s]

radar = Radar(
    fcar=10e9,  # carrier frequency [Hz]
    tx_power=1e3,  # transmit power [W]
    tx_gain=10 ** (30 / 10),  # transmit antenna gain [linear], 30 dB
    rx_gain=10 ** (30 / 10),  # receive antenna gain [linear], 30 dB
    op_temp=290,  # operating temperature [K]
    sample_rate=2 * bw,  # Nyquist sampling of the waveform bandwidth
    noise_factor=10 ** (8 / 10),  # receiver noise factor [linear], 8 dB
    total_losses=10 ** (8 / 10),  # total system losses [linear], 8 dB
    prf=20e3,  # pulse repetition frequency [Hz]
    dwell_time=1.5e-3,  # 30 pulses
)

waveform = lfm_waveform(bw, T=T, chirp_up_down=1)  # 1 us up-chirp
range_bin = c.C / (2 * radar.sample_rate)  # 7.5 m


def pinned_target(range: float, snr_db: float) -> Target:
    """Target on the nearest range-bin centre with the given single-pulse SNR."""
    range = round(range / range_bin) * range_bin
    snr_per_m2 = snr_range_eqn(
        radar.tx_power,
        radar.tx_gain,
        radar.rx_gain,
        1.0,
        c.C / radar.fcar,
        range,
        bw,
        radar.noise_factor,
        radar.total_losses,
        radar.op_temp,
        bw * T,
    )
    return Target(range=range, range_rate=40, rcs=10 ** (snr_db / 10) / snr_per_m2)


def snr_profile(targets: list[Target], n_pulses: int, seed: int = 0):
    """NCI range profile of the first ``n_pulses`` pulses, in per-pulse SNR units."""
    r_axis, rtm = rdm.gen_rtm(radar, waveform, [Return(target=t) for t in targets], seed=seed)
    snr_rtm = rdm.rtm_to_snr(rtm, radar, waveform)
    return r_axis, rdm.noncoherent_integrate(snr_rtm[:, :n_pulses])


# -- Range profiles: one 3 dB target, 1 vs N pulses --
pfa = 1e-6
target = pinned_target(4e3, snr_db=3)
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
for ax, n in zip(axes, [1, 10, 30]):
    r_axis, profile = snr_profile([target], n)
    threshold = detection.threshold_factor_nci(pfa, n)
    ax.plot(r_axis * 1e-3, 10 * np.log10(profile), lw=0.8)
    ax.axhline(
        10 * np.log10(threshold), color="k", ls="--", label=f"threshold, $P_{{fa}}$={pfa:.0e}"
    )
    ax.axvline(target.range * 1e-3, color="r", ls=":", label="target")
    ax.set_ylabel("NCI power [dB]")
    ax.set_title(f"N = {n}")
    ax.grid(True)
axes[0].legend(loc="lower right")
axes[-1].set_xlabel("Range [km]")
fig.suptitle("Range profile, 3 dB single-pulse SNR target")
fig.tight_layout()

# -- Monte Carlo: Pfa and Pd vs theory --
# Four targets at each per-pulse SNR, 40 bins (300 m) apart, so every trial gives
# 4 Pd samples per SNR and about half the profile as noise-only bins.
pfa = 1e-3
n_trials = 100
snr_levels_db = [0, 3, 6, 9, 12]
targets = [
    pinned_target(600 + 300 * i, snr_db) for i, snr_db in enumerate(np.repeat(snr_levels_db, 4))
]
target_bins = np.array([round(t.range / range_bin) - 1 for t in targets])  # r_axis is 1-indexed
target_snr_db = np.repeat(snr_levels_db, 4)

r_axis, _ = snr_profile(targets, 1)
dist_to_target = np.min(np.abs(np.arange(len(r_axis))[:, None] - target_bins[None, :]), axis=1)
noise_bins = dist_to_target > 20  # outside every target's matched-filter output

n_list = [1, 10]
hits = {n: np.zeros(len(targets)) for n in n_list}
target_power = {n: np.zeros(len(targets)) for n in n_list}
false_alarms = {n: 0 for n in n_list}
noise_mean = {n: 0.0 for n in n_list}
for seed in range(n_trials):
    _, rtm = rdm.gen_rtm(radar, waveform, [Return(target=t) for t in targets], seed=seed)
    snr_rtm = rdm.rtm_to_snr(rtm, radar, waveform)
    for n in n_list:
        profile = rdm.noncoherent_integrate(snr_rtm[:, :n])
        detected = profile > detection.threshold_factor_nci(pfa, n)
        hits[n] += detected[target_bins]
        target_power[n] += profile[target_bins] / n_trials
        false_alarms[n] += detected[noise_bins].sum()
        noise_mean[n] += profile[noise_bins].mean() / n_trials

n_noise = n_trials * noise_bins.sum()
print(f"Pfa = {pfa:.0e}, {n_trials} trials, {n_noise} noise-only samples per N")
for n in n_list:
    print(f"\nN = {n}:")
    print(f"  noise mean       = {noise_mean[n]:.3f} (theory {n})")
    print(f"  measured Pfa     = {false_alarms[n] / n_noise:.2e}")
    print("  SNR_1   mean power (theory)   Pd measured (theory)")
    for snr_db in snr_levels_db:
        sel = target_snr_db == snr_db
        snr = 10 ** (snr_db / 10)
        pd_meas = hits[n][sel].sum() / (n_trials * sel.sum())
        pd_theory = detection.pd_swerling0_nci(snr, pfa, n)
        mean_meas = target_power[n][sel].mean()
        print(
            f"  {snr_db:3d} dB  {mean_meas:7.1f} ({n * (1 + snr):7.1f})"
            f"      {pd_meas:.3f} ({pd_theory:.3f})"
        )

fig, ax = plt.subplots(figsize=(8, 5))
snr_axis_db = np.arange(-5, 16, 0.2)
for n in n_list:
    pd_theory = detection.pd_swerling0_nci(10 ** (snr_axis_db / 10), pfa, n)
    (line,) = ax.plot(snr_axis_db, pd_theory, label=f"theory, N={n}")
    pd_meas = [hits[n][target_snr_db == s].sum() / (n_trials * 4) for s in snr_levels_db]
    ax.plot(snr_levels_db, pd_meas, "o", color=line.get_color(), label=f"simulated, N={n}")
ax.set_xlabel("Single-pulse SNR [dB]")
ax.set_ylabel("$P_d$")
ax.set_title(f"NCI detection, Swerling 0 ($P_{{fa}}$ = {pfa:.0e})")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True)
fig.tight_layout()

# -- Integration gain: NCI vs coherent, Pd = 0.9 --
print(f"\nIntegration gain for Pd = 0.9, Pfa = {pfa:.0e}:")
print("           NCI      coherent  NCI loss")
snr_single = detection.required_snr(0.9, pfa)
for n in [1, 10, 30]:
    gain_nci = snr_single - detection.required_snr_nci(0.9, pfa, n)
    gain_coh = 10 * np.log10(n)
    print(f"  N = {n:2d}: {gain_nci:5.2f} dB  {gain_coh:5.2f} dB  {gain_coh - gain_nci:.2f} dB")

plt.show()
