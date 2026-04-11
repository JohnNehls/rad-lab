#!/usr/bin/env python
"""STAP demonstration: clutter suppression with adaptive processing.

Simulates an airborne ULA pulse-Doppler radar with:
- A moving platform that creates angle-Doppler coupled ground clutter
- Two moving targets at the broadside look direction, at different ranges and
  radial velocities

Compares conventional beamforming + Doppler FFT against STAP (Sample Matrix
Inversion) to show how adaptive processing suppresses clutter while preserving
target detections.  Both the conventional and adaptive range-Doppler maps are
formed at a single look direction (broadside); the targets are placed there so
they are spatially matched.  A target well off the look angle is attenuated and
its apparent Doppler is distorted -- explored in the validation exercise.

Key takeaways:
- Ground clutter from an airborne radar has a Doppler shift that depends on
  the look angle (clutter ridge), so clutter from off-boresight angles leaks
  through the array sidelobes across a wide band of Doppler -- a simple Doppler
  notch cannot reject it.
- STAP jointly filters in angle and Doppler, placing adaptive nulls on the
  clutter ridge while steering toward the target, suppressing the clutter that
  a fixed beamform-then-FFT leaves in the sidelobes and cleaning the noise
  floor around the targets.
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

# -- Targets (at the broadside look direction, angle = 0) --
# Unambiguous range: c/(2*prf) = 15 km, unambiguous rdot: ±75 m/s
# Target 1: 7 km, closing at 50 m/s, 20 dBsm
# Target 2: 10 km, receding at 30 m/s, 10 dBsm
return_list = [
    Return(target=Target(range=7e3, range_rate=-50, rcs=100, angle=0)),
    Return(target=Target(range=10e3, range_rate=30, rcs=10, angle=0)),
]

# -- Run STAP simulation --
print("## STAP Demonstration ##")
print(f"  Platform velocity: {platform_velocity} m/s")
print(f"  Array: {n_elements} elements, λ/2 spacing")
print(f"  Targets: {len(return_list)}")

# Guard bins must exceed the target's range-compressed extent (~ the pulse
# length in samples) so a strong target's compression sidelobes do not leak
# into the covariance training data and get nulled along with the clutter.
waveform.set_sample(radar.sample_rate)
n_guard = len(waveform.pulse_sample) + 5

result = stap.gen(
    radar,
    waveform,
    return_list,
    el_pos=el_pos,
    platform_velocity=platform_velocity,
    cnr=40,  # aggregate clutter-to-noise power ratio (~16 dB) at each range bin
    n_clutter_patches=180,
    steer_angle=0,
    plot=True,
    n_guard=n_guard,
    diagonal_load=1.0,
)

print("\nConventional RDM peak:", f"{np.abs(result['conventional']).max():.2e}")
print("Adaptive RDM peak:", f"{np.abs(result['adaptive']).max():.2e}")

plt.show()
