#!/usr/bin/env python
"""Barker code autocorrelation sidelobe comparison.

Compare the autocorrelation (matched-filter output) of an uncoded pulse to
every standard Barker code length. Barker codes are prized because their
autocorrelation sidelobes are at most 1/N of the peak, giving clean detection
with minimal range ambiguity.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab.waveform_helpers import matchfilter_with_waveform
from rad_lab.waveform import uncoded_pulse, barker_coded_pulse, BARKER_DICT
from rad_lab.waveform_helpers import zeropad_waveform


print("#############################################")
print("Problem 2: Barker sidelobe check code example")
print("#############################################\n")

print("The sidelobes do not follow the table")
print("\tconsider the ypeak/nchips comment in the problem")

# -- Waveform parameters --
BW = 4e6  # waveform bandwidth [Hz]
sampleRate = 16e6  # sample rate [Hz]
Npad = 50  # zero-pad length for cleaner correlation display

# -- Create the uncoded (reference) pulse and zero-pad it --
tu, mag_u = uncoded_pulse(sampleRate, BW)
tu, mag_u = zeropad_waveform(tu, mag_u, Npad)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("S3P2 Barker sidelobe check")

# Left plot: time-domain pulse shapes
ax[0].plot(tu, mag_u, label="uncoded")
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("pulse [v]")
ax[0].set_title("normalized pulse")

# Right plot: autocorrelation (matched filter of pulse with itself)
iu, conv_u = matchfilter_with_waveform(mag_u, mag_u)
conv_u = abs(conv_u)
ax[1].plot(iu, conv_u, "-x", label="uncoded")
ax[1].set_xlabel("index offset")
ax[1].set_ylabel("magnitude")
ax[1].set_title("xcorrelation")

# -- Loop over all Barker code lengths and overlay results --
# For each code, measure the peak-to-max-sidelobe ratio.  Barker codes have
# sidelobes of at most 1 chip vs a peak of N chips, so the ratio should be ~N.
print("peak / max sidelobe of the autocorrelation (theory: nchips)")
samples_per_chip = int(sampleRate / BW)
for nChip in BARKER_DICT.keys():
    t_b, mag_b = barker_coded_pulse(sampleRate, BW, nChip)
    ax[0].plot(t_b, mag_b, label=f"barker {nChip}")

    t_b, mag_b = zeropad_waveform(t_b, mag_b, Npad)
    ib, conv_b = matchfilter_with_waveform(mag_b, mag_b)
    conv_b = abs(conv_b)
    ax[1].plot(ib, conv_b, label=f"barker {nChip}")

    # Sidelobes are everything outside the mainlobe (one chip wide either side)
    i_peak = np.argmax(conv_b)
    sidelobes = np.concatenate([
        conv_b[: i_peak - samples_per_chip],
        conv_b[i_peak + samples_per_chip + 1 :],
    ])
    print(f"\tbarker {nChip:2d}: {conv_b[i_peak] / sidelobes.max():5.2f}")

for a in ax:
    a.grid()
    a.legend()

plt.show()
