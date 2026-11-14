#!/usr/bin/env python
"""MTI on range profiles.

Non-coherently integrate the 9_2 clutter scene into range profiles with 0-,
1-, and 2-pulse cancellers.  With no Doppler FFT, the canceller is the only
clutter filter.  Then measure the false-alarm rate of the NCI threshold on
noise-only MTI output.

Notes:
  - Without MTI the clutter-to-noise ratio (CNR) falls from ~30 dB at 1 km
    to ~7 dB at 6 km (median ~12 dB) and buries a 1 m^2 target that the
    RDMs of 9_2 showed clearly.  The canceller drops the clutter to the
    noise floor and the target emerges.
  - Residual clutter after MTI is CNR minus the improvement factor (27 dB
    order 1, 51 dB order 2).  Over most of the band that is below the
    noise, so order 2 adds little at the target.  Near 1 km the CNR exceeds
    the order-1 improvement factor and residue crosses the threshold; order
    2 clears it.
  - The canceller also filters the target: its gain relative to noise is
    ``|H(f_d)|^2 / noise_gain``, printed below.  A target at a blind speed
    (here 300 m/s) is cancelled along with the clutter.
  - The canceller correlates noise from pulse to pulse (order 1 outputs share
    an input sample with each neighbour).  The NCI sum keeps the right mean
    but gets a heavier tail than the chi-squared statistic that
    `threshold_factor_nci` assumes, so the measured Pfa exceeds the design
    value.  Integrating every (order + 1)-th pulse uses only independent
    outputs and restores it, at the cost of integration gain.  CFAR, which
    sets the threshold from neighbouring cells, is the usual cure.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm, mti, detection, Radar, Target, Return, Clutter, lfm_waveform
from rad_lab import constants as c

bw = 10e6  # waveform bandwidth [Hz]

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
    dwell_time=5e-3,  # coherent processing interval [s], 100 pulses
)

waveform = lfm_waveform(bw, T=1.0e-6, chirp_up_down=1)  # 1 us up-chirp

clutter = Clutter(
    range_start=1e3,
    range_stop=6e3,
    gamma=10 ** (-20 / 10),  # -20 dB constant gamma
    grazing_angle=np.deg2rad(2),
    azimuth_beamwidth=np.deg2rad(2),
    range_rate_std=3.0,  # internal-motion spread [m/s]
)

# -- Targets on range-bin centres: one moving, one at the first blind speed --
blind_speed = mti.blind_speeds(radar.prf, radar.fcar, n=1)[0]  # 300 m/s
target = Target(range=4005, range_rate=120, rcs=1)
blind_target = Target(range=5002.5, range_rate=blind_speed, rcs=1)
return_list = [Return(target=target), Return(target=blind_target), clutter]

# -- Range profiles with MTI off, order 1, and order 2 --
pfa = 1e-6
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 8))
for ax, order in zip(axes, [0, 1, 2]):
    r_axis, rtm = rdm.gen_rtm(radar, waveform, return_list, mti_order=order)
    n_pulses = rtm.shape[1]
    # Divide by N so the noise floor is 0 dB in every panel.
    profile = rdm.noncoherent_integrate(rdm.rtm_to_snr(rtm, radar, waveform, order)) / n_pulses

    i_t = np.argmin(abs(r_axis - target.range))
    i_b = np.argmin(abs(r_axis - blind_target.range))
    background = (
        (r_axis > clutter.range_start + 200)
        & (r_axis < clutter.range_stop - 200)
        & (abs(r_axis - target.range) > 200)
        & (abs(r_axis - blind_target.range) > 200)
    )
    floor = np.median(profile[background])
    doppler = 2 * target.range_rate * radar.fcar / c.C
    target_gain = mti.frequency_response(order, radar.prf, np.array([doppler]))[0]
    print(f"MTI order {order}:")
    print(f"  clutter background / noise  = {10 * np.log10(floor):5.1f} dB")
    print(f"  target / background         = {10 * np.log10(profile[i_t] / floor):5.1f} dB")
    print(f"  blind target / background   = {10 * np.log10(profile[i_b] / floor):5.1f} dB")
    print(
        f"  target gain vs noise, theory = {10 * np.log10(target_gain / mti.noise_gain(order)):5.1f} dB"
    )

    threshold = detection.threshold_factor_nci(pfa, n_pulses) / n_pulses
    ax.plot(r_axis * 1e-3, 10 * np.log10(profile), lw=0.8)
    ax.axhline(
        10 * np.log10(threshold), color="k", ls="--", label=f"NCI threshold, $P_{{fa}}$={pfa:.0e}"
    )
    ax.axvline(
        target.range * 1e-3, color="r", ls=":", label=f"target, {target.range_rate:.0f} m/s"
    )
    ax.axvline(
        blind_target.range * 1e-3,
        color="m",
        ls=":",
        label=f"target, {blind_speed:.0f} m/s (blind)",
    )
    ax.set_ylabel("Mean power per pulse [dB SNR]")
    ax.set_title("no MTI" if order == 0 else f"order-{order} MTI")
    ax.grid(True)
axes[0].legend(loc="upper right", fontsize="small")
axes[-1].set_xlabel("Range [km]")
fig.suptitle("NCI range profiles of a cluttered scene")
fig.tight_layout()

# -- False-alarm rate on noise-only MTI output --
pfa = 1e-3
n_trials = 20
print(f"\nmeasured Pfa on noise only (design {pfa:.0e}, {n_trials} trials):")
print("  order   every pulse   every (order+1)-th pulse")
for order in [0, 1, 2]:
    false_alarms = {1: 0, order + 1: 0}
    n_cells = 0
    for seed in range(n_trials):
        _, rtm = rdm.gen_rtm(radar, waveform, [], seed=seed, mti_order=order)
        snr_rtm = rdm.rtm_to_snr(rtm, radar, waveform, order)
        n_cells += snr_rtm.shape[0]
        for stride in false_alarms:
            pulses = snr_rtm[:, ::stride]
            threshold = detection.threshold_factor_nci(pfa, pulses.shape[1])
            false_alarms[stride] += np.sum(rdm.noncoherent_integrate(pulses) > threshold)
    every, strided = false_alarms[1] / n_cells, false_alarms[order + 1] / n_cells
    print(f"  {order}       {every:.2e}      {strided:.2e}")

plt.show()
