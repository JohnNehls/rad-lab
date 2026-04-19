#!/usr/bin/env python
"""Spotlight SAR example: three point targets with beam-pattern weighting.

Demonstrates spotlight SAR by steering the antenna toward a scene centre
and using a longer synthetic aperture than the stripmap example.  The
longer aperture yields finer cross-range resolution at the cost of a
larger datacube.

Compared to stripmap_point_targets.py:
  - aperture_length increased from 50 m to 200 m  (4× more pulses)
  - scene_center set to the middle target's position
  - beamwidth set from a notional 0.5 m antenna: λ/D ≈ 0.06 rad

Cross-range resolution comparison (at ~5.8 km slant range):
  Stripmap:  λR/(2L) = 0.03 × 5831 / (2 × 50)  ≈ 1.75 m
  Spotlight: λR/(2L) = 0.03 × 5831 / (2 × 200) ≈ 0.44 m  (~4× finer)
"""

import matplotlib.pyplot as plt
from rad_lab import sar, SarRadar, SarTarget, lfm_waveform

# -- Waveform --
bw = 5e6  # bandwidth [Hz] → range resolution = c/(2*bw) = 30 m
waveform = lfm_waveform(bw, T=10e-6, chirp_up_down=1)

# -- Scene centre (the point the antenna tracks) --
scene_center = [0, 3e3, 0]

# -- SAR system parameters (spotlight mode) --
sar_radar = SarRadar(
    fcar=10e9,  # carrier frequency [Hz] → λ = 0.03 m
    tx_power=1e3,  # transmit power [W]
    tx_gain=10 ** (30 / 10),  # transmit gain [linear], 30 dB
    rx_gain=10 ** (30 / 10),  # receive gain [linear], 30 dB
    op_temp=290,  # operating temperature [K]
    sample_rate=2 * bw,  # 10 MHz → 1250 range bins per PRI
    noise_factor=10 ** (8 / 10),  # noise factor [linear], 8 dB
    total_losses=10 ** (8 / 10),  # system losses [linear], 8 dB
    prf=8000,  # PRF [Hz]
    platform_velocity=100,  # platform speed [m/s]
    aperture_length=200,  # synthetic aperture [m] → 16000 pulses
    platform_altitude=5e3,  # altitude [m]
    scene_center=scene_center,  # spotlight: steer beam here
    beamwidth=0.06,  # one-way 3-dB beamwidth [rad] (≈ λ/D, D ≈ 0.5 m)
)

# -- Point targets in the scene --
targets = [
    SarTarget(position=[-5, 0.1e3, 0], rcs=10),
    SarTarget(position=[0, 3e3, 0], rcs=10),
    SarTarget(position=[5, 5e3, 0], rcs=10),
]

# -- Generate the focused SAR image --
cross_range, slant_range, total_image = sar.gen(
    sar_radar, waveform, targets, seed=0, plot=True, debug=False
)

plt.show()
