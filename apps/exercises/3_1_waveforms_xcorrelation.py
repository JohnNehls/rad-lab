#!/usr/bin/env python
"""Waveform cross-correlation exercises.

Generate uncoded, Barker-coded, random-coded, and LFM pulses, then visualize
each pulse's time-domain shape, power spectral density, and auto-correlation.

Key takeaway: BPSK bandwidth is set by the chip width (same as a single chip
pulse), but the PSDs become noisier with more chips. LFM spreads energy
uniformly across the bandwidth.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab.waveform_helpers import plot_pulse_and_spectrum, plot_pulse_and_xcorrelation
from rad_lab.waveform import uncoded_pulse, barker_coded_pulse, random_coded_pulse, lfm_pulse

np.random.seed(0)  # reproducible random-coded pulse

# -- Common waveform parameters (normalized units) --
sample_rate = 10
bw = 1

# -- Uncoded pulse: simplest waveform (rectangular envelope) --
print("## uncoded example ##")
t_u, mag_u = uncoded_pulse(sample_rate, bw, normalize=False)
plot_pulse_and_spectrum(t_u, mag_u, f"S3P1 uncoded pulse: {sample_rate=}, {bw=}", n_pad=500)
plot_pulse_and_xcorrelation(t_u, mag_u, f"uncoded pulse: {sample_rate=}, {bw=}")

# -- Barker-coded pulse: phase-coded with low autocorrelation sidelobes --
print("## Barker example ##")
nchip = 7
t_b, mag_b = barker_coded_pulse(sample_rate, bw, nchip, normalize=False)
plot_pulse_and_spectrum(
    t_b, mag_b, f"S3P1 Barker coded pulse: {nchip=}, {sample_rate=}, {bw=}", n_pad=500
)
plot_pulse_and_xcorrelation(t_b, mag_b, f"Barker coded pulse: {nchip=}, {sample_rate=}, {bw=}")

# -- Random-coded pulse: phase-coded with random ±1 chips --
print("## random code example  ##")
nchip = 7
t_r, mag_r = random_coded_pulse(sample_rate, bw, nchip, normalize=False)
plot_pulse_and_spectrum(
    t_r, mag_r, f"S3P1 random coded pulse: {nchip=}, {sample_rate=}, {bw=}", n_pad=500
)
plot_pulse_and_xcorrelation(
    t_r, mag_r, f"S3P1 random coded pulse: {nchip=}, {sample_rate=}, {bw=}"
)

# -- LFM (chirp) pulse: frequency sweeps linearly over the bandwidth --
print("## LFM example ##")
T = 2
chirp_up_down = 1  # +1 = up-chirp, -1 = down-chirp
t_lfm, mag_lfm = lfm_pulse(sample_rate, bw, T, chirp_up_down, normalize=False)
fig, ax = plot_pulse_and_spectrum(
    t_lfm, mag_lfm, f"S3P1 LFM pulse: {chirp_up_down=}, {T=}, {sample_rate=}, {bw=}", n_pad=500
)

plot_pulse_and_xcorrelation(
    t_lfm, mag_lfm, f"S3P1 LFM pulse: {chirp_up_down=}, {T=}, {sample_rate=}, {bw=}"
)
plt.show()
