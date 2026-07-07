#!/usr/bin/env python
"""Datacube processing test.

Build a raw RF datacube, inject a tone at a known range bin and Doppler
frequency, then apply Doppler processing (FFT across slow time) to verify
the signal appears in the correct range-Doppler cell.

The datacube has two axes:
  - Fast time (rows): samples within a single PRI — maps to range.
  - Slow time (columns): one sample per pulse — maps to Doppler frequency.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab.rf_datacube import data_cube, doppler_process
from rad_lab.constants import PI


print("##########################")
print("Test datacube processing")
print("##########################")

# -- Radar timing parameters --
fs = 20e6  # sampling frequency [Hz] (sets range resolution)
PRF = 100e3  # pulse repetition frequency [Hz]
Np = 256  # number of pulses in the CPI

# -- Create an empty datacube and inject a test signal --
dc = data_cube(fs, PRF, Np)  # shape: (range_bins, Np)
dt_pulse = 1 / PRF  # time between pulses [s]
t_ar = np.arange(Np) * dt_pulse  # slow-time axis

# Place a complex sinusoid at range bin 98, oscillating at PRF/4 in slow time.
# After Doppler processing, this should appear at Doppler bin = Np/4.
dc[98, :] = np.exp(2j * PI * PRF / 4 * t_ar)

# -- Visualize raw vs processed datacube --
fig, ax = plt.subplots(1, 2)
fig.suptitle("test datacube processing")

# Left: raw datacube (before Doppler processing)
ax[0].set_title("unprocessed datacube")
im = ax[0].imshow(abs(dc), origin="lower")
ax[0].set_xlabel("slow time [PRI]")
ax[0].set_ylabel("fast time [fs]")
fig.colorbar(im, ax=ax[0])

# Doppler process: FFT along slow-time axis (modifies dc in-place)
f_ax, r_ax = doppler_process(dc, fs)

# Right: range-Doppler map — energy should be concentrated in one cell
ax[1].set_title("processed datacube")
mesh = ax[1].pcolormesh(f_ax * 1e-6, r_ax, abs(dc))
ax[1].set_xlabel("frequency [MHz]")
ax[1].set_ylabel("range [m]")
fig.colorbar(mesh, ax=ax[1])
plt.tight_layout()

# -- Verify the RDM peak landed in the expected range-Doppler cell --
peak_range_bin, peak_doppler_bin = np.unravel_index(np.argmax(abs(dc)), dc.shape)
print(f"RDM peak: range bin {peak_range_bin}, Doppler {f_ax[peak_doppler_bin] * 1e-6:.3f} MHz")
print(f"expected: range bin 98, Doppler {PRF / 4 * 1e-6:.3f} MHz")
if peak_range_bin != 98 or not np.isclose(f_ax[peak_doppler_bin], PRF / 4):
    raise Exception("RDM processed incorrectly")

plt.show()
