#!/usr/bin/env python
"""Window function comparison on a rectangular pulse.

Apply Chebyshev, Blackman-Harris, and Taylor windows to an uncoded pulse and
compare the resulting spectra. Windowing reduces spectral sidelobes at the
cost of widening the mainlobe (lower range resolution).

Key observations:
  - Unwindowed (rectangular): narrowest mainlobe, highest sidelobes (~-13 dB).
  - Chebyshev: equi-ripple sidelobes at the specified level (60 dB here).
  - Blackman-Harris: very low sidelobes, but widest mainlobe.
  - Taylor: compromise — low near-in sidelobes with moderate mainlobe widening.
"""

from scipy import signal
import matplotlib.pyplot as plt
from rad_lab.waveform import uncoded_pulse
from rad_lab.waveform_helpers import plot_pulse_and_spectrum


print("##########################")
print("Test windowing")
print("##########################")

# -- Pulse parameters --
fs = 100e6  # sampling frequency [Hz]
bw = 11e6  # pulse bandwidth [Hz]

# -- Generate an uncoded (rectangular) pulse --
t_u, mag_u = uncoded_pulse(fs, bw)

# -- Create window functions (same length as the pulse) --
chwin = signal.windows.chebwin(mag_u.size, 60)  # Chebyshev, 60 dB sidelobe suppression
bhwin = signal.windows.blackmanharris(mag_u.size)  # Blackman-Harris
tywin = signal.windows.taylor(mag_u.size)  # Taylor

# -- Apply each window by element-wise multiplication --
mag_chwin = chwin * mag_u
mag_bhwin = bhwin * mag_u
mag_tywin = tywin * mag_u

# -- Plot pulse shape and spectrum for each case --
n_pad = 4001  # zero-pad length for smooth spectral estimate

print("uncoded")
plot_pulse_and_spectrum(t_u, mag_u, "unfiltered pulse", n_pad, spec_dec=True)
print("Chebyshev")
plot_pulse_and_spectrum(t_u, mag_chwin, "chwin filtered pulse", n_pad, spec_dec=True)
print("Blackman-Harris")
plot_pulse_and_spectrum(t_u, mag_bhwin, "bhwin filtered pulse", n_pad, spec_dec=True)
print("Taylor (should be smaller BW)")
plot_pulse_and_spectrum(t_u, mag_tywin, "tywin filtered pulse", n_pad, spec_dec=True)

plt.show()
