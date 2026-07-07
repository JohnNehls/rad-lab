#!/usr/bin/env python
"""Monopulse angle estimation: time-domain vs frequency-domain comparison.

Extends the basic monopulse SNR test (7_5) to show that monopulse angle
estimation works identically in both domains. Four methods are compared:

  1. Time-domain monopulse ratio (sum/difference of element signals)
  2. Time-domain phase-only estimate (only valid at baseband, not RF passband)
  3. Frequency-domain monopulse ratio (applied at the peak FFT bin)
  4. Frequency-domain phase-only estimate

Key result: the monopulse ratio gives the same accuracy in time and frequency
domains. The phase-only method works at baseband but fails at RF passband.
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

# -- Sweep SNR and compute angle estimation error for all four methods --
snr_db_list = np.arange(-15, 50, step=5)

err_time_ratio_mean = []  # time-domain monopulse ratio
err_time_ratio_std = []
err_time_phase_mean = []  # time-domain phase-only
err_freq_ratio = []  # frequency-domain monopulse ratio
err_freq_phase = []  # frequency-domain phase-only

for snr_db in snr_db_list:
    # -- Simulate received signal at each array element --
    snr_volt_scale = 10 ** (snr_db / 20)
    received_signals = []
    for sv in steer_vec:
        if BASEBAND:
            # Baseband: signal is a constant phasor (no time oscillation)
            received_signals.append(snr_volt_scale * sv + unity_variance_complex_noise(N_samples))
        else:
            # RF passband: signal oscillates at 'freq' Hz
            received_signals.append(
                snr_volt_scale * sv * signal_ar + unity_variance_complex_noise(N_samples)
            )
    plt.plot(np.real(received_signals[0]), label=f"snr={snr_db}")

    # -- Method 1: time-domain monopulse ratio (sample-by-sample) --
    dx = array_pos[1] - array_pos[0]  # element separation [wavelengths]
    measured_theta = mp.monopulse_angle_deg(received_signals[0], received_signals[1], dx)
    measured_error = abs(measured_theta - tgt_angle)
    err_time_ratio_mean.append(np.mean(measured_error))
    err_time_ratio_std.append(np.std(measured_error))

    # -- Method 2: time-domain phase-only (only valid at baseband) --
    angle_est_naive = (np.angle(received_signals[0]) - np.angle(received_signals[1])) / (np.pi)
    err_time_phase_mean.append(np.mean(abs(np.rad2deg(angle_est_naive) - tgt_angle)))

    # -- Method 3: frequency-domain monopulse ratio --
    # Window and FFT each element's signal, then apply monopulse at the peak bin
    f_received_signals = []
    for sig in received_signals:
        f_received_signals.append(np.fft.fft(sig * signal.windows.chebwin(sig.size, 60)))

    f_max_index = np.argmax(np.abs(f_received_signals[0]))
    f_measured_theta = mp.monopulse_angle_at_peak_deg(
        f_received_signals[0], f_received_signals[1], dx
    )
    f_measured_error = abs(f_measured_theta - tgt_angle)
    err_freq_ratio.append(f_measured_error)

    # -- Method 4: frequency-domain phase-only --
    f_angle_est_naive = (np.angle(f_received_signals[0]) - np.angle(f_received_signals[1]))[
        f_max_index
    ] / (np.pi)
    err_freq_phase.append(abs(np.rad2deg(f_angle_est_naive) - tgt_angle))


# -- Summarize all four methods: errors should fall with SNR, and the
# monopulse-ratio methods should agree between time and frequency domains --
print(f"angle error [deg] vs SNR (true angle = {tgt_angle} deg, BASEBAND={BASEBAND})")
print("\tSNR [dB]   time ratio   time phase   freq ratio   freq phase")
for i, snr_db in enumerate(snr_db_list):
    print(
        f"\t{snr_db:8d}   {err_time_ratio_mean[i]:10.3f}   {err_time_phase_mean[i]:10.3f}"
        f"   {err_freq_ratio[i]:10.3f}   {err_freq_phase[i]:10.3f}"
    )

# -- Plot received signals at each SNR --
plt.legend()
plt.title("Noisy signal for Each SNR [dB]")
plt.xlabel("sample")
plt.ylabel("amplitude [v]")
plt.grid()

# -- Plot angle estimation error: all four methods compared --
fig, axs = plt.subplots(1, 2)
fig.suptitle("Monopulse Angle Estimation")
axs[0].plot(snr_db_list, err_time_ratio_mean, label="time domain (mean)")
axs[0].plot(snr_db_list, err_time_phase_mean, "o", label="time domain (mean) phase")
axs[0].plot(snr_db_list, err_freq_ratio, label="freq domain (@max)")
axs[0].plot(snr_db_list, err_freq_phase, "o", label="freq domain (@max) phase")
axs[0].set_title("Angle Error")
axs[0].legend()
axs[1].plot(snr_db_list, err_time_ratio_std)
axs[1].set_title(r"$\sigma$ Angle Error")

for ax in axs:
    ax.grid()
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Angle [Deg]")

plt.tight_layout()
plt.show()
