#!/usr/bin/env python
"""Stripmap SAR example: three point targets at different positions.

Demonstrates the basic SAR workflow:
  1. Define SAR system parameters (SarRadar) and an LFM waveform.
  2. Place point targets (SarTarget) in the scene.
  3. Call sar.gen() to simulate the collection and focus the image.

The focused image should show three distinct peaks at the target positions.
Cross-range resolution depends on the synthetic aperture length and range;
range resolution depends on the waveform bandwidth.

Parameter design summary:
  - λ = 0.03 m (10 GHz carrier)
  - Azimuth Nyquist: pulse_spacing = v/prf = 0.0125 m < λ/2 = 0.015 m
  - Range resolution: c/(2*bw) = 30 m
  - Cross-range resolution: λ*R/(2*L) ≈ 1.5–2.1 m (varies with slant range)
  - Datacube: 1250 range bins × 4000 pulses
  - Targets at slant ranges ~5.0, 5.8, 7.1 km (separated by ~800+ m, ≫ 30 m res)
    and along-track positions −5, 0, +5 m (separated by 5 m, > 1.5–2.1 m res)
"""

import matplotlib.pyplot as plt
from rad_lab import sar, SarRadar, SarTarget, lfm_waveform

# -- Waveform --
bw = 5e6  # bandwidth [Hz] → range resolution = c/(2*bw) = 30 m
waveform = lfm_waveform(bw, T=10e-6, chirp_up_down=1)  # time-bandwidth product = 50

# -- SAR system parameters --
# Key constraint: prf ≥ 2*v/λ = 6667 Hz (azimuth Nyquist)
sar_radar = SarRadar(
    fcar=10e9,  # carrier frequency [Hz] → λ = 0.03 m
    tx_power=1e3,  # transmit power [W]
    tx_gain=10 ** (30 / 10),  # transmit gain [linear], 30 dB
    rx_gain=10 ** (30 / 10),  # receive gain [linear], 30 dB
    op_temp=290,  # operating temperature [K]
    sample_rate=2 * bw,  # 10 MHz → 1250 range bins per PRI
    noise_factor=10 ** (8 / 10),  # noise factor [linear], 8 dB
    total_losses=10 ** (8 / 10),  # system losses [linear], 8 dB
    prf=8000,  # PRF [Hz], > 2*v/λ = 6667 Hz
    platform_velocity=100,  # platform speed [m/s]
    aperture_length=50,  # synthetic aperture [m] → 4000 pulses
    platform_altitude=5e3,  # altitude [m]
)

# -- Point targets in the scene --
# Convention: [x_along_track, y_cross_track, z_altitude]
# Slant ranges: ~5.0 km, ~5.8 km, ~7.1 km (separated by ~800+ m, ≫ 30 m res)
# Along-track: −5 m, 0 m, +5 m (separated by 5 m, > 1.5–2.1 m res)
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
