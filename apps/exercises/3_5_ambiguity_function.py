#!/usr/bin/env python
"""Ambiguity function exercises.

Compute and display the ambiguity function for three waveform types: uncoded,
Barker-coded, and LFM.  Each waveform's ambiguity surface and zero-delay /
zero-Doppler cuts are plotted to illustrate the range-Doppler resolution
tradeoffs.

Key takeaways:
- Uncoded pulse: narrow in delay (good range resolution for a given bandwidth)
  but wide in Doppler — poor velocity resolution.  The ambiguity surface is a
  "thumbtack" shape.
- Barker-coded pulse: similar mainlobe width to uncoded (set by chip bandwidth)
  but with lower autocorrelation sidelobes — better clutter rejection.
- LFM pulse: a diagonal ridge showing range-Doppler coupling.  Narrow in both
  dimensions, but a target's apparent range shifts with its Doppler frequency.
"""

import matplotlib.pyplot as plt
from rad_lab.waveform import uncoded_pulse, barker_coded_pulse, lfm_pulse
from rad_lab.ambiguity import ambiguity_function, plot_ambiguity, plot_zero_cuts

# -- Common parameters --
sample_rate = 100e3  # Hz
bw = 10e3  # Hz
fd_max = 20e3  # Hz — Doppler axis extent (±20 kHz)
tau_lim_us = (-1500, 1500)  # delay-axis view limits [µs] — fits Barker-13 envelope
fd_lim_khz = (-20, 20)  # Doppler-axis view limits [kHz]


def _apply_limits(fig_ax_ambig, fig_ax_cuts):
    _, ax = fig_ax_ambig
    ax.set_xlim(*tau_lim_us)
    ax.set_ylim(*fd_lim_khz)
    _, (ax_tau, ax_fd) = fig_ax_cuts
    ax_tau.set_xlim(*tau_lim_us)
    ax_fd.set_xlim(*fd_lim_khz)


# -- Uncoded pulse --
print("## Uncoded pulse ##")
_, pulse_uncoded = uncoded_pulse(sample_rate, bw, normalize=False)
tau, fd, af = ambiguity_function(pulse_uncoded, sample_rate, fd_max=fd_max)
_apply_limits(
    plot_ambiguity(tau, fd, af, title="Ambiguity Function — Uncoded Pulse"),
    plot_zero_cuts(tau, fd, af, title="Zero Cuts — Uncoded Pulse"),
)

# -- Barker-13 coded pulse --
print("## Barker-13 coded pulse ##")
_, pulse_barker = barker_coded_pulse(sample_rate, bw, nchips=13, normalize=False)
tau, fd, af = ambiguity_function(pulse_barker, sample_rate, fd_max=fd_max)
_apply_limits(
    plot_ambiguity(tau, fd, af, title="Ambiguity Function — Barker-13"),
    plot_zero_cuts(tau, fd, af, title="Zero Cuts — Barker-13"),
)

# -- LFM pulse (time-bandwidth product = 100) --
print("## LFM pulse ##")
T_lfm = 1e-3  # 1 ms pulse → TBP = bw * T = 10
_, pulse_lfm = lfm_pulse(sample_rate, bw, T_lfm, chirp_up_down=1, normalize=False)
tau, fd, af = ambiguity_function(pulse_lfm, sample_rate, fd_max=fd_max)
_apply_limits(
    plot_ambiguity(tau, fd, af, title="Ambiguity Function — LFM (up-chirp)"),
    plot_zero_cuts(tau, fd, af, title="Zero Cuts — LFM (up-chirp)"),
)

plt.show()
