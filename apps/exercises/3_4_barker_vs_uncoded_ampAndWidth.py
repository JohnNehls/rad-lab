#!/usr/bin/env python
"""Compare Barker-13 vs uncoded pulse amplitude and matched-filter width.

Both pulses use the same chip bandwidth, so the Barker-13 pulse is 13x longer
in time. After matched filtering, the Barker pulse produces a sharper peak
(same mainlobe width as uncoded) but with 13x higher processing gain, while
the uncoded pulse has a triangular autocorrelation with no sidelobes.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab.waveform_helpers import matchfilter_with_waveform, zeropad_waveform
from rad_lab.waveform import uncoded_pulse, barker_coded_pulse


print("##############################")
print("Problem 4: Compare Barker 13 to uncoded pulse")
print("##############################")

# -- Waveform parameters --
BW = 4e6  # waveform bandwidth [Hz]
sampleRate = 16e6  # sample rate [Hz]
SNR = 20  # [dB]

# -- Generate both pulses and scale to the same SNR --
tb, mag_b = barker_coded_pulse(sampleRate, BW, 13)
mag_b_s = 10 ** (SNR / 20) * mag_b

tu, mag_u = uncoded_pulse(sampleRate, BW)
tu, mag_u = zeropad_waveform(tu, mag_u, 50)  # zero-pad for visual comparison
mag_u_s = 10 ** (SNR / 20) * mag_u

# -- Plot: time-domain pulses and their matched-filter outputs --
fig, ax = plt.subplots(1, 2)
fig.suptitle("Sec 2 prob 4 : compare uncoded to Barker 13 pulse")

# Left: time-domain waveforms — Barker is longer but same amplitude
ax[0].plot(tu, mag_u_s, "-o", label="uncoded")
ax[0].plot(tb, mag_b_s, "-x", label="barker13")
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("pulse amplitude [v]")
ax[0].legend()

# Right: matched-filter output — Barker peak is ~13x taller (processing gain)
iu, conv_u = matchfilter_with_waveform(mag_u_s, mag_u)
ib, conv_b = matchfilter_with_waveform(mag_b_s, mag_b)

# The Barker pulse spreads the same (unit-normalized) energy over 13 chips, so
# it is 13x longer with 1/sqrt(13) the amplitude, yet the matched-filter peaks
# and mainlobe widths match — same detection energy and range resolution.
n_u = int((abs(mag_u) > 0).sum())
n_b = int((abs(mag_b) > 0).sum())
print(f"pulse length:    uncoded={n_u} samples, barker13={n_b} samples")
print(f"pulse amplitude: uncoded={abs(mag_u_s).max():.2f} v, barker13={abs(mag_b_s).max():.2f} v")
peak_u = abs(conv_u).max()
peak_b = abs(conv_b).max()
width_u = int((abs(conv_u) >= peak_u / np.sqrt(2)).sum())  # -3 dB width [samples]
width_b = int((abs(conv_b) >= peak_b / np.sqrt(2)).sum())
print(f"matched-filter peaks: uncoded={peak_u:.1f} barker13={peak_b:.1f}")
print(f"-3 dB mainlobe width: uncoded={width_u} samples, barker13={width_b} samples")
ax[1].plot(iu, conv_u, "-o", label="uncoded")
ax[1].plot(ib, conv_b, "-x", label="barker13")
ax[1].set_xlabel("index shift")
ax[1].set_ylabel("matched filter")
ax[1].legend()
plt.tight_layout()
for a in ax:
    a.grid()

plt.show()
