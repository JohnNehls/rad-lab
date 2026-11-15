#!/usr/bin/env python
"""Staggered PRI against MTI blind speeds.

Cycle the PRI from pulse to pulse (`Radar.pri_stagger`) so a target cancelled
by one PRI is passed by the next.  Compare a constant PRI with a narrow
(23:27:24:26) and a wide (4:5:6) stagger, all at a 20 kHz mean PRF: the
canceller's velocity response, MTI range profiles of the 09_1 scene (with a
brighter, 5 m^2 target at the constant-PRI blind speed), and the clutter
improvement factor.

Notes:
  - The canceller taps are unchanged; each output simply sees different pulse
    spacings.  A target is fully cancelled only when its Doppler is a multiple
    of every PRF at once, so the first blind speed moves out by
    mean(ratios) / gcd(ratios): 25x (7.5 km/s) for the narrow stagger, 5x
    (1.5 km/s) for the wide one.
  - The old blind speeds become notches rather than nulls.  Their depth is set
    by the spread of the ratios: about -11 dB for the narrow stagger and
    -3 dB for the wide one.  More spread fills the notches but pulls the first
    blind speed in, so choosing the ratios is a trade between the two.
  - Clutter rejection suffers too.  Taps that sum to zero still null zero
    Doppler for any spacing, so order 1 barely changes, but [1, -2, 1] loses
    its double zero at DC on uneven spacing and order 2 drops by 4-9 dB.
  - The range gate shrinks to the shortest PRI (6.9 km narrow, 6 km wide), so
    the clutter band here stops at 5.5 km.  Echoes from beyond the gate would
    land at a different range on every pulse and escape the canceller; they
    are not part of this scene.
  - The Doppler FFT needs evenly spaced pulses, so a staggered radar has no
    RDM here (`rdm.gen` refuses it).  Staggering suits MTI-only processing such
    as these range profiles.
"""

import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm, mti, Radar, Target, Return, Clutter, lfm_waveform
from rad_lab import pulse_doppler_radar as pdr

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
    prf=20e3,  # (mean) pulse repetition frequency [Hz]
    dwell_time=5e-3,  # 100 pulses
)

waveform = lfm_waveform(bw, T=1.0e-6, chirp_up_down=1)  # 1 us up-chirp

clutter = Clutter(
    range_start=1e3,
    range_stop=5.5e3,  # inside the shortest range gate (6 km, wide stagger)
    gamma=10 ** (-20 / 10),  # -20 dB constant gamma
    grazing_angle=np.deg2rad(2),
    azimuth_beamwidth=np.deg2rad(2),
    range_rate_std=3.0,  # internal-motion spread [m/s]
)

staggers = {"constant PRI": None, "narrow 23:27:24:26": (23, 27, 24, 26), "wide 4:5:6": (4, 5, 6)}
blind_speed = mti.blind_speeds(radar.prf, radar.fcar, n=1)[0]  # 300 m/s, constant PRI

# -- Targets on range-bin centres: one moving, one at the constant-PRI blind speed --
target = Target(range=4005, range_rate=120, rcs=1)
blind_target = Target(range=5002.5, range_rate=blind_speed, rcs=5)
return_list = [Return(target=target), Return(target=blind_target), clutter]

# -- Order-1 canceller velocity response, relative to its noise gain --
order = 1
fig, axes = plt.subplots(2, 1, figsize=(8, 7))
for ax, v_max in zip(axes, [1600, 8000]):
    range_rates = np.linspace(0, v_max, 4001)
    freqs = pdr.frequency_delta_doppler(range_rates, radar.fcar)
    for label, stagger in staggers.items():
        response = mti.frequency_response(order, radar.prf, freqs, stagger) / mti.noise_gain(order)
        with np.errstate(divide="ignore"):
            ax.plot(range_rates, 10 * np.log10(response), lw=1, label=label)
    ax.set_ylim(-40, 5)
    ax.set_ylabel("Gain vs. noise [dB]")
    ax.grid(True)
axes[0].axvline(blind_speed, color="k", ls=":", lw=0.8)
axes[0].legend(loc="lower right")
axes[-1].set_xlabel("Range Rate [m/s]")
fig.suptitle(f"Order-{order} canceller response, {radar.prf / 1e3:.0f} kHz mean PRF")
fig.tight_layout()

# -- MTI range profiles --
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 8))
for ax, (label, stagger) in zip(axes, staggers.items()):
    staggered_radar = dataclasses.replace(radar, pri_stagger=stagger)
    r_axis, rtm = rdm.gen_rtm(staggered_radar, waveform, return_list, mti_order=order)
    n_pulses = rtm.shape[1]
    # Divide by N so the noise floor is 0 dB in every panel.
    snr_rtm = rdm.rtm_to_snr(rtm, staggered_radar, waveform, order)
    profile = rdm.noncoherent_integrate(snr_rtm) / n_pulses
    i_t = np.argmin(abs(r_axis - target.range))
    i_b = np.argmin(abs(r_axis - blind_target.range))

    ax.plot(r_axis * 1e-3, 10 * np.log10(profile), lw=0.8)
    ax.axvline(target.range * 1e-3, color="r", ls=":", label=f"{target.range_rate:.0f} m/s")
    ax.axvline(blind_target.range * 1e-3, color="m", ls=":", label=f"{blind_speed:.0f} m/s")
    ax.set_ylabel("Mean power per pulse [dB SNR]")
    ax.set_title(f"{label}, range gate {r_axis[-1] * 1e-3:.1f} km")
    ax.grid(True)
    print(f"{label}: target {10 * np.log10(profile[i_t]):.1f} dB, ", end="")
    print(f"{blind_speed:.0f} m/s target {10 * np.log10(profile[i_b]):.1f} dB above noise")
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Range [km]")
fig.suptitle(f"Order-{order} MTI range profiles")
fig.tight_layout()

# -- Blind speeds and improvement factor (noiseless, clutter only) --
quiet_radar = dataclasses.replace(radar, op_temp=0)


def clutter_power(stagger: tuple[int, ...] | None, order: int) -> float:
    """Mean clutter power per range bin and pulse after an order-``order`` canceller."""
    staggered = dataclasses.replace(quiet_radar, pri_stagger=stagger)
    r_axis, rtm = rdm.gen_rtm(staggered, waveform, [clutter], mti_order=order)
    band = (r_axis > clutter.range_start) & (r_axis < clutter.range_stop)
    return np.mean(np.abs(rtm[band]) ** 2)


print("\n                       first blind  gain at   improvement factor (measured vs. theory)")
print("                       speed [m/s]  300 m/s   order 1          order 2")
for label, stagger in staggers.items():
    first_blind = mti.blind_speeds(radar.prf, radar.fcar, n=1, pri_stagger=stagger)[0]
    gain_300 = mti.frequency_response(
        1, radar.prf, np.array([pdr.frequency_delta_doppler(blind_speed, radar.fcar)]), stagger
    )[0] / mti.noise_gain(1)
    gain_300_db = f"{10 * np.log10(gain_300):5.1f} dB" if gain_300 > 1e-10 else "null"
    line = f"  {label:20s} {first_blind:7.0f}  {gain_300_db:>8s}"
    power_in = clutter_power(stagger, 0)
    for o in [1, 2]:
        measured = mti.noise_gain(o) / (clutter_power(stagger, o) / power_in)
        theory = mti.improvement_factor(o, radar.prf, radar.fcar, clutter.range_rate_std, stagger)
        line += f"  {10 * np.log10(measured):5.1f} vs {10 * np.log10(theory):5.1f}"
    print(line)

plt.show()
