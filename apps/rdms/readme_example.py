#!/usr/bin/env python
"""Minimal range-Doppler map example (used in the README).

Steps:
  1. Define a radar system (carrier, power, gains, timing).
  2. Choose a waveform (Barker-13 coded pulse).
  3. Define a target (range, range-rate, RCS).
  4. Generate the RDM — this builds the datacube, adds noise, applies the
     matched filter, and Doppler-processes to produce the range-Doppler map.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm, Radar, Target, Return, barker_coded_waveform

# -- Define the radar --
radar = Radar(
    fcar=10e9,  # carrier frequency [Hz]
    tx_power=1e3,  # transmit power [W]
    tx_gain=10 ** (30 / 10),  # transmit gain [linear], 30 dB
    rx_gain=10 ** (30 / 10),  # receive gain [linear], 30 dB
    op_temp=290,  # noise temperature [K]
    sample_rate=20e6,  # ADC sample rate [Hz]
    noise_factor=10 ** (8 / 10),  # noise factor [linear], 8 dB
    total_losses=10 ** (8 / 10),  # system losses [linear], 8 dB
    prf=200e3,  # pulse repetition frequency [Hz]
    dwell_time=2e-3,  # coherent processing interval [s]
)

# -- Define the waveform: Barker-13 BPSK with 10 MHz bandwidth --
waveform = barker_coded_waveform(10e6, nchips=13)

# -- Define a single skin return --
return_list = [Return(target=Target(range=0.5e3, range_rate=1.0e3, rcs=1))]

# -- Generate and display the range-Doppler map --
rdot_axis, r_axis, datacube = rdm.gen(radar, waveform, return_list)

# The RDM peak should land in the target's range/range-rate cell
peak_r, peak_rdot = np.unravel_index(np.argmax(abs(datacube)), datacube.shape)
print(f"RDM peak: range = {r_axis[peak_r] * 1e-3:.2f} km (target at 0.50 km)")
print(f"          range rate = {rdot_axis[peak_rdot] * 1e-3:.2f} km/s (target at 1.00 km/s)")

plt.show()
