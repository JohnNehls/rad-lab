#!/usr/bin/env python
"""Coherent Doppler processing of staggered pulses.

The slow-time FFT needs evenly spaced pulses, but coherent integration does
not: `rdm.doppler_bank` phase-aligns and sums the pulses at their actual
transmit times.  Compare a constant PRI, the 10_2 narrow and wide staggers,
and a random PRI jitter: their Doppler responses, the coherent gain, and
range-Doppler maps of the 10_2 scene with and without MTI first.

Notes:
  - The coherent gain is the full N (20 dB for 100 pulses) for every pulse
    train: a target at the filter's Doppler adds in phase whatever the spacing.
  - Stagger does not remove Doppler ambiguity, it reshapes it.  A repeating
    cycle of M PRIs is itself a periodic pulse train, so a target also
    appears at multiples of prf/M with levels set by the cycle.  The pulses
    also stay close to the mean-PRF grid, so the alias at the mean PRF (300
    m/s here) stays strong: -0.2 dB for the narrow stagger, -1.6 dB for the
    wide one, whose ghosts every 100 m/s range from -11 to -2 dB.
  - Random jitter has no repeating cycle, so no comb of ghosts, but it still
    keeps a partial alias at the mean PRF (-5 dB here, since every PRI is
    within 20% of the mean).  The rest of the ambiguous energy spreads into
    a sidelobe floor, -23 dB median here (about 1/N for N pulses), that no
    window can lower.
  - Clutter goes through the same response.  The ridge at zero Doppler
    reappears at every ghost, and under jitter as a raised floor across all
    Doppler.  Between the ridge and its first constant-PRI alias (30-270
    m/s) the 99th-percentile level is 6.6 dB for a constant PRI, which is
    pure noise (ln 100 for unit-mean exponential power), but ~20 dB for wide
    stagger and jitter.  An MTI canceller first removes most of the clutter
    before it can leak, which is why MTI followed by Doppler filtering is
    the usual order.
  - For the constant PRI the 300 m/s cell is the aliased clutter ridge, not
    the target, which MTI then cancels along with the clutter.
  - After MTI the remaining structure is the targets' own response: ghosts
    every 100 m/s for the wide stagger, a streak across Doppler for jitter.
  - The Chebyshev taper on pulse index controls sidelobes of the mainlobe only.
    It cannot touch ghosts or the jitter floor, which come from the sampling
    pattern, not from the finite dwell.
"""

import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm, mti, Radar, Target, Return, Clutter, lfm_waveform
from rad_lab import constants as c
from rad_lab import pulse_doppler_radar as pdr
from rad_lab.range_equation import snr_range_eqn
from rad_lab.rf_datacube import taper

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
    prf=20e3,  # (mean) pulse repetition frequency [Hz]
    dwell_time=5e-3,  # 100 pulses
)

waveform = lfm_waveform(bw, T=T, chirp_up_down=1)  # 1 us up-chirp

clutter = Clutter(
    range_start=1e3,
    range_stop=5.5e3,  # inside the shortest range gate
    gamma=10 ** (-20 / 10),  # -20 dB constant gamma
    grazing_angle=np.deg2rad(2),
    azimuth_beamwidth=np.deg2rad(2),
    range_rate_std=3.0,  # internal-motion spread [m/s]
)


def random_jitter(n_pulses: int, low: int, high: int, mean: int, seed: int = 1) -> tuple:
    """Random PRI ratios in [low, high] with an exact integer mean (one per pulse)."""
    rng = np.random.default_rng(seed)
    ratios = rng.integers(low, high + 1, n_pulses)
    while (excess := ratios.sum() - mean * n_pulses) != 0:
        movable = np.flatnonzero(ratios > low if excess > 0 else ratios < high)
        ratios[rng.choice(movable)] -= np.sign(excess)
    return tuple(int(r) for r in ratios)


staggers = {
    "constant PRI": None,
    "narrow 23:27:24:26": (23, 27, 24, 26),
    "wide 4:5:6": (4, 5, 6),
    "random jitter 20-30": random_jitter(radar.n_pulses, 20, 30, mean=25),
}


# -- Doppler response of each pulse train (a unit target at zero Doppler) --
range_rates = np.linspace(0, 1500, 6001)
freqs = pdr.frequency_delta_doppler(range_rates, radar.fcar)
fig, axes = plt.subplots(len(staggers), 1, sharex=True, sharey=True, figsize=(8, 9))
print("strongest Doppler response, 10 m/s off the target to below the first blind speed:")
for ax, (label, stagger) in zip(axes, staggers.items()):
    times = dataclasses.replace(radar, pri_stagger=stagger).pulse_tx_times
    response = np.abs(rdm.doppler_bank(np.ones((1, len(times))), times, freqs)[0])
    response_db = 20 * np.log10(response / len(times))
    first_blind = mti.blind_speeds(radar.prf, radar.fcar, n=1, pri_stagger=stagger)[0]
    away = (range_rates > 10) & (range_rates < first_blind - 10)
    i_peak = np.argmax(response_db * away - 1e3 * ~away)
    floor = np.median(response_db[away])
    print(
        f"  {label:20s} {response_db[i_peak]:6.1f} dB at {range_rates[i_peak]:6.0f} m/s,"
        f" median {floor:6.1f} dB"
    )
    ax.plot(range_rates, response_db, lw=0.6)
    ax.set_title(label)
    ax.set_ylabel("[dB]")
    ax.set_ylim(-50, 3)
    ax.grid(True)
axes[-1].set_xlabel("Range-rate offset from target [m/s]")
fig.suptitle("Doppler response of the pulse train (100 pulses, no taper)")
fig.tight_layout()

# -- Coherent gain: noiseless target, filter at its exact Doppler --
target = Target(range=4005, range_rate=120, rcs=1)
snr_1 = snr_range_eqn(
    radar.tx_power,
    radar.tx_gain,
    radar.rx_gain,
    target.rcs,
    c.C / radar.fcar,
    target.range,
    bw,
    radar.noise_factor,
    radar.total_losses,
    radar.op_temp,
    bw * T,
)
print(f"\ncoherent gain, {radar.n_pulses} pulses (theory {10 * np.log10(radar.n_pulses):.1f} dB):")
for label, stagger in staggers.items():
    quiet = dataclasses.replace(radar, op_temp=0, pri_stagger=stagger)
    r_axis, rtm = rdm.gen_rtm(quiet, waveform, [Return(target=target)])
    snr_rtm = rdm.rtm_to_snr(rtm, radar, waveform)  # normalise by the noisy radar's noise
    i_r = np.argmin(abs(r_axis - target.range))
    target_freq = np.array([pdr.frequency_delta_doppler(target.range_rate, radar.fcar)])
    y = rdm.doppler_bank(snr_rtm[i_r : i_r + 1], quiet.pulse_tx_times, target_freq)
    # Output SNR = |y|^2 / N (noise variance N); gain = output SNR / single-pulse SNR.
    gain = np.abs(y[0, 0]) ** 2 / radar.n_pulses / snr_1
    print(f"  {label:20s} {10 * np.log10(gain):.1f} dB")

# -- Range-Doppler maps of the 10_2 scene, without and with MTI --
blind_target = Target(range=5002.5, range_rate=300, rcs=5)
return_list = [Return(target=target), Return(target=blind_target), clutter]
rdot_axis = np.arange(-600, 601, 3.0)  # ~ one Doppler resolution cell (1 / dwell) per step
map_freqs = pdr.frequency_delta_doppler(rdot_axis, radar.fcar)
map_staggers = ["constant PRI", "wide 4:5:6", "random jitter 20-30"]

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8))
print(
    "\nrange-Doppler maps [dB SNR]:  120 m/s tgt  300 m/s tgt  clutter leak 30-270 m/s (99th pct)"
)
for col, label in enumerate(map_staggers):
    staggered = dataclasses.replace(radar, pri_stagger=staggers[label])
    for row, order in enumerate([0, 1]):
        r_axis, rtm = rdm.gen_rtm(staggered, waveform, return_list, mti_order=order)
        weights = taper(rtm.shape[1], "chebyshev", {"at": 60})
        weights /= np.sqrt(np.sum(weights**2))  # unit noise variance out of the bank
        snr_rtm = rdm.rtm_to_snr(rtm, staggered, waveform, order) * weights
        rdm_map = rdm.doppler_bank(snr_rtm, staggered.pulse_tx_times[order:], map_freqs)
        snr_db = 20 * np.log10(np.abs(rdm_map) + 1e-12)

        i_t = np.argmin(abs(r_axis - target.range)), np.argmin(abs(rdot_axis - target.range_rate))
        i_b = (
            np.argmin(abs(r_axis - blind_target.range)),
            np.argmin(abs(rdot_axis - blind_target.range_rate)),
        )
        band = (r_axis > clutter.range_start + 200) & (r_axis < clutter.range_stop - 200)
        band &= np.abs(r_axis - target.range) > 200
        off_ridge = (np.abs(rdot_axis) > 30) & (np.abs(rdot_axis) < 270)  # inside +/- PRF/2
        leak = np.percentile(snr_db[np.ix_(band, off_ridge)], 99)
        name = f"{label}, {'no MTI' if order == 0 else 'order-1 MTI'}"
        print(f"  {name:36s} {snr_db[i_t]:6.1f}       {snr_db[i_b]:6.1f}       {leak:6.1f}")

        ax = axes[row, col]
        mesh = ax.pcolormesh(rdot_axis, r_axis * 1e-3, snr_db, vmin=0, vmax=50)
        ax.set_title(name, fontsize="medium")
        if row == 1:
            ax.set_xlabel("Range Rate [m/s]")
        if col == 0:
            ax.set_ylabel("Range [km]")
fig.colorbar(mesh, ax=axes, label="SNR [dB]")
fig.suptitle("Doppler bank on staggered pulses (60 dB Chebyshev taper)")

plt.show()
