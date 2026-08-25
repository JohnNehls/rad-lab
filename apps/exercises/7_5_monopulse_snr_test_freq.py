#!/usr/bin/env python
"""Monopulse angle estimation: time-domain vs frequency-domain comparison.

Extend the basic monopulse SNR test (7_5) to show that monopulse angle
estimation works identically in both domains. Compare four methods:

  1. Time-domain monopulse ratio (sum/difference of element signals)
  2. Time-domain phase-only estimate (only valid at baseband, not RF passband)
  3. Frequency-domain monopulse ratio (applied at the peak FFT bin)
  4. Frequency-domain phase-only estimate (survives at RF: the peak bin
     demodulates the tone)

There is no pulse train here: the record is a single dwell of N complex
samples of one tone.  "Coherent integration" means summing those N samples,
which gives ~10*log10(N) of SNR gain.  Both domains do exactly that same sum
before forming the ratio -- the time domain adds the samples directly (at RF,
after demodulating the tone), and the frequency domain reads the peak FFT bin,
which is that identical sum written in the frequency domain (bit-for-bit equal
when unwindowed).  Errors are averaged over Monte Carlo noise trials at each
SNR.

Key takeaways:

- The monopulse ratio gives the same accuracy in time and frequency domains,
  because both apply the one coherent sum over the N-sample dwell (the
  time-domain sum equals the peak FFT bin).
- The naive phase-only estimate reads the angle straight from the
  inter-element phase difference.  It works at baseband, but the time-domain
  version fails at RF passband (summing the spinning carrier destroys the
  signal); the frequency-domain version survives because reading the peak bin
  demodulates the tone.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import rad_lab.uniform_linear_arrays as ula
import rad_lab.monopulse as mp
from rad_lab.noise import unity_variance_complex_noise


plt.rcParams["text.usetex"] = True

# -- Signal configuration --
BASEBAND = True  # True: signal at DC; False: oscillating at RF
np.random.seed(100)
N_samples = 1000
time_ar = np.linspace(0, 10, N_samples)
freq = 1  # [Hz]
signal_ar = np.exp(1j * 2 * np.pi * freq * time_ar)  # used only when BASEBAND=False

# -- Two-element array --
tgt_angle = -5  # true target angle [deg]
array_pos = np.array([-1 / 4, 1 / 4])  # element positions [wavelengths]
steer_vec = ula.steering_vector(array_pos, tgt_angle)

# -- Sweep SNR; at each SNR run several noise realizations so every method's
# mean and spread are estimated from the same Monte Carlo trials --
snr_db_list = np.arange(-15, 50, step=5)
N_TRIALS = 500  # noise realizations per SNR

# Reference for coherently integrating the RATIO methods: summing each element
# against it adds up all N samples of the dwell, exactly the sum the
# frequency-domain peak bin computes.  At baseband it is unity (a plain
# coherent sum); at RF it demodulates the tone before summing.
reference = np.ones(N_samples) if BASEBAND else signal_ar
dx = array_pos[1] - array_pos[0]  # element separation [wavelengths]

err_time_ratio_mean = []  # time-domain monopulse ratio
err_time_ratio_std = []
err_freq_ratio_mean = []  # frequency-domain monopulse ratio
err_freq_ratio_std = []
err_time_phase_mean = []  # time-domain phase-only
err_freq_phase_mean = []  # frequency-domain phase-only

for snr_db in snr_db_list:
    snr_volt_scale = 10 ** (snr_db / 20)
    e_time_ratio, e_freq_ratio, e_time_phase, e_freq_phase = [], [], [], []

    for trial in range(N_TRIALS):
        # -- Simulate the received signal at each array element --
        received_signals = []
        for sv in steer_vec:
            # Baseband: constant phasor.  RF passband: oscillates at 'freq' Hz.
            tgt = snr_volt_scale * sv * (1.0 if BASEBAND else signal_ar)
            received_signals.append(tgt + unity_variance_complex_noise(N_samples))

        if trial == N_TRIALS - 1:  # one representative noisy trace per SNR
            plt.plot(np.real(received_signals[0]), label=f"snr={snr_db}")

        # -- Method 1: time-domain monopulse ratio, coherently integrated --
        # Sum all N samples of each element against the reference (one coherent
        # integration over the dwell), then form the ratio on the integrated
        # pair.  This is the time-domain equivalent of the peak FFT bin.
        a = np.sum(received_signals[0] * np.conj(reference))
        b = np.sum(received_signals[1] * np.conj(reference))
        e_time_ratio.append(abs(mp.monopulse_angle_deg(a, b, dx) - tgt_angle))

        # -- Method 2: time-domain phase-only (only valid at baseband) --
        # Estimate the angle straight from the inter-element phase difference.
        # Elements at +/-1/4 wavelength have steering phases +/-(pi/2)*sin(theta),
        # so their difference is pi*sin(theta); dividing by pi recovers
        # sin(theta), which for small angles is theta itself.  Unlike the ratio
        # (Im(delta/sum)), this needs the signal already at baseband: it uses the
        # RAW (non-demodulated) coherent sum, so at RF the spinning carrier
        # averages to ~zero when summed and the phase difference degenerates into
        # noise -- the estimate fails.
        a_raw, b_raw = received_signals[0].sum(), received_signals[1].sum()
        phase_est = np.rad2deg((np.angle(a_raw) - np.angle(b_raw)) / np.pi)
        e_time_phase.append(abs(phase_est - tgt_angle))

        # -- Method 3: frequency-domain monopulse ratio at the peak bin --
        # The windowed FFT integrates the record; the peak bin holds the
        # target, so this mirrors the coherently integrated time estimate.
        f_signals = [np.fft.fft(s * signal.windows.chebwin(s.size, 60)) for s in received_signals]
        peak = np.argmax(np.abs(f_signals[0]))
        f_theta = mp.monopulse_angle_at_peak_deg(f_signals[0], f_signals[1], dx)
        e_freq_ratio.append(abs(f_theta - tgt_angle))

        # -- Method 4: frequency-domain phase-only at the peak bin --
        # The same phase difference, but reading the peak bin (at the tone's
        # frequency) demodulates the carrier, so unlike Method 2 this survives
        # at RF passband.
        f_phase = np.rad2deg((np.angle(f_signals[0]) - np.angle(f_signals[1]))[peak] / np.pi)
        e_freq_phase.append(abs(f_phase - tgt_angle))

    err_time_ratio_mean.append(np.mean(e_time_ratio))
    err_time_ratio_std.append(np.std(e_time_ratio))
    err_freq_ratio_mean.append(np.mean(e_freq_ratio))
    err_freq_ratio_std.append(np.std(e_freq_ratio))
    err_time_phase_mean.append(np.mean(e_time_phase))
    err_freq_phase_mean.append(np.mean(e_freq_phase))


# -- Summarize all four methods: errors should fall with SNR, and the
# monopulse-ratio methods should agree between time and frequency domains --
print(f"angle error [deg] vs SNR (true angle = {tgt_angle} deg, BASEBAND={BASEBAND})")
print("\tSNR [dB]   time ratio   time phase   freq ratio   freq phase")
for i, snr_db in enumerate(snr_db_list):
    print(
        f"\t{snr_db:8d}   {err_time_ratio_mean[i]:10.3f}   {err_time_phase_mean[i]:10.3f}"
        f"   {err_freq_ratio_mean[i]:10.3f}   {err_freq_phase_mean[i]:10.3f}"
    )

# -- Plot received signals at each SNR --
plt.legend()
plt.title("Noisy signal for Each SNR [dB]")
plt.xlabel("sample")
plt.ylabel("amplitude [v]")
plt.grid()

# -- Plot angle estimation error: all four methods compared --
# Color encodes the domain (time vs frequency); style encodes the method
# (ratio = line, phase-only = dots).
TIME_COLOR = "tab:blue"
FREQ_COLOR = "tab:orange"

fig, axs = plt.subplots(1, 2)
fig.suptitle("Monopulse Angle Estimation")
axs[0].plot(snr_db_list, err_time_ratio_mean, color=TIME_COLOR, label="time ratio")
axs[0].plot(snr_db_list, err_time_phase_mean, "o", color=TIME_COLOR, label="time phase")
axs[0].plot(snr_db_list, err_freq_ratio_mean, color=FREQ_COLOR, label="freq ratio")
axs[0].plot(snr_db_list, err_freq_phase_mean, "o", color=FREQ_COLOR, label="freq phase")
axs[0].set_title("Angle Error")
axs[0].legend()
axs[1].plot(snr_db_list, err_time_ratio_std, color=TIME_COLOR, label="time ratio")
axs[1].plot(snr_db_list, err_freq_ratio_std, color=FREQ_COLOR, label="freq ratio")
axs[1].set_title(r"$\sigma$ Angle Error")
axs[1].legend()

for ax in axs:
    ax.grid()
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Angle [Deg]")

plt.tight_layout()
plt.show()
