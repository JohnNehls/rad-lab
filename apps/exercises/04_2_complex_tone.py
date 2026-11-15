#!/usr/bin/env python
"""Complex tone windowing exercise.

Generate a complex tone (single-frequency sinusoid) and show how spectral
leakage from the rectangular window is suppressed by applying a Chebyshev
window before computing the spectrum.
"""

from scipy import signal
import matplotlib.pyplot as plt
from rad_lab.waveform import complex_tone_pulse
from rad_lab.waveform_helpers import plot_pulse_and_spectrum


# -- Signal parameters --
fs = 200e6  # sampling frequency [Hz]
bw = 11e6  # pulse bandwidth [Hz]

# -- Generate a complex tone at fs/8 (25 MHz) --
t_s, mag_s = complex_tone_pulse(fs, bw, fs / 8)

# -- Apply a Chebyshev window to suppress spectral leakage --
chwin = signal.windows.chebwin(mag_s.size, 60)  # 60 dB sidelobe suppression
mag_chwin_s = chwin * mag_s

# -- Compare spectra: without vs with window --
print("complex tone without window")
plot_pulse_and_spectrum(t_s, mag_s, "complex tone without filter", n_pad=1024)
print("complex tone with chebwin window")
plot_pulse_and_spectrum(t_s, mag_chwin_s, "complex tone with chwin filter", n_pad=1024)

plt.show()
