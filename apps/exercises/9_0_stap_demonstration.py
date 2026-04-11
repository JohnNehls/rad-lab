#!/usr/bin/env python
"""STAP demonstration: clutter suppression with adaptive processing.

Simulates an airborne ULA pulse-Doppler radar with:
- A moving platform that creates angle-Doppler coupled ground clutter
- Two moving targets at different angles and velocities

Compares conventional beamforming + Doppler FFT against STAP (Sample Matrix
Inversion) to show how adaptive processing suppresses clutter while preserving
target detections.

Key takeaways:
- Ground clutter from an airborne radar has a Doppler shift that depends on
  the look angle (clutter ridge), making it impossible to reject with a
  simple Doppler notch filter.
- Conventional processing (beamform then FFT) cannot separate targets from
  clutter when they share the same Doppler or angle.
- STAP jointly filters in angle and Doppler, placing adaptive nulls on the
  clutter ridge while steering toward the target, dramatically improving
  target visibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import stap
from rad_lab.pulse_doppler_radar import Radar
from rad_lab.waveform import lfm_waveform
from rad_lab.returns import Target, Return

# -- Waveform --
bw = 1e6  # Hz
waveform = lfm_waveform(bw, T=10e-6, chirp_up_down=1)

# -- Radar --
radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (20 / 10),
    rx_gain=10 ** (20 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (5 / 10),
    total_losses=10 ** (3 / 10),
    prf=10e3,
    dwell_time=1.6e-3,  # 16 pulses
)

# -- Array: 8-element ULA, half-wavelength spacing --
n_elements = 8
el_pos = np.arange(n_elements) * 0.5 - (n_elements - 1) * 0.25  # centred

# -- Platform velocity (creates clutter Doppler coupling) --
platform_velocity = 100  # m/s

# -- Targets --
# Unambiguous range: c/(2*prf) = 15 km, unambiguous rdot: ±75 m/s
# Target 1: 20 deg off broadside, closing at 50 m/s, 20 dBsm
# Target 2: -10 deg off broadside, receding at 30 m/s, 10 dBsm
return_list = [
    Return(target=Target(range=7e3, range_rate=-50, rcs=100, angle=20)),
    Return(target=Target(range=10e3, range_rate=30, rcs=10, angle=-10)),
]

# -- Run STAP simulation --
print("## STAP Demonstration ##")
print(f"  Platform velocity: {platform_velocity} m/s")
print(f"  Array: {n_elements} elements, λ/2 spacing")
print(f"  Targets: {len(return_list)}")

result = stap.gen(
    radar,
    waveform,
    return_list,
    el_pos=el_pos,
    platform_velocity=platform_velocity,
    cnr=40,  # 40x noise power per clutter patch
    n_clutter_patches=180,
    steer_angle=0,
    plot=True,
    n_guard=3,
    diagonal_load=1e-2,
)

print("\nConventional RDM peak:", f"{np.abs(result['conventional']).max():.2e}")
print("Adaptive RDM peak:", f"{np.abs(result['adaptive']).max():.2e}")

plt.show()
