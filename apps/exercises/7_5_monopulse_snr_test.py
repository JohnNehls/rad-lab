#!/usr/bin/env python
"""Monopulse angle estimation accuracy vs SNR.

Simulate a two-element array receiving a signal from a known angle, add noise
at varying SNR levels, and measure how accurately the monopulse ratio estimates
the target angle.

Monopulse works by comparing the signals at two array elements (sum and
difference channels). The ratio of difference to sum gives an angle estimate
that is independent of signal amplitude.

Key observations:
  - Estimation accuracy improves dramatically with SNR.
  - Accuracy also depends on the true target angle (closer to boresight = better).
"""

import numpy as np
import matplotlib.pyplot as plt
import rad_lab.uniform_linear_arrays as ula
import rad_lab.monopulse as mp
from rad_lab.noise import unity_variance_complex_noise

np.random.seed(0)  # reproducible noise realizations

plt.rcParams["text.usetex"] = True

# -- Generate a test signal (single-frequency complex sinusoid) --
N_samples = 1000
time_ar = np.linspace(0, 10, N_samples)
freq = 1  # [Hz]
signal_ar = np.exp(1j * 2 * np.pi * freq * time_ar)

# -- Define the two-element linear array --
tgt_angle = 2  # true target angle [deg] from boresight
array_pos = np.array([-1 / 4, 1 / 4])  # element positions [wavelengths]
steer_vec = ula.steering_vector(array_pos, tgt_angle)  # phase shifts per element

# -- Sweep SNR and compute monopulse angle estimation error --
snr_db_list = np.arange(30, step=3)

error_mean_list = []
error_std_list = []
for snr_db in snr_db_list:
    snr_volt_scale = 10 ** (snr_db / 20)

    # Simulate received signal at each element: scaled signal + noise
    received_signals = []
    for sv in steer_vec:
        received_signals.append(
            snr_volt_scale * sv * signal_ar + unity_variance_complex_noise(N_samples)
        )
    plt.plot(np.real(received_signals[0]), label=f"snr={snr_db}")

    # Estimate angle from the monopulse ratio (sample-by-sample)
    dx = array_pos[1] - array_pos[0]  # element separation [wavelengths]
    measured_theta = mp.monopulse_angle_deg(received_signals[0], received_signals[1], dx)

    # Compute estimation error statistics across all samples
    measured_error = abs(measured_theta - tgt_angle)
    error_mean_list.append(np.mean(measured_error))
    error_std_list.append(np.std(measured_error))

# -- Summarize: error statistics should fall monotonically with SNR --
print(f"monopulse angle error vs SNR (true angle = {tgt_angle} deg)")
print("\tSNR [dB]   mean [deg]   std [deg]")
for snr_db, err_mean, err_std in zip(snr_db_list, error_mean_list, error_std_list):
    print(f"\t{snr_db:8d}   {err_mean:10.3f}   {err_std:9.3f}")

# -- Plot the received signals at each SNR --
plt.legend()
plt.title("Noisy signal for Each SNR [dB]")
plt.xlabel("sample")
plt.ylabel("amplitude [v]")
plt.grid()

# -- Plot angle estimation error vs SNR --
fig, axs = plt.subplots(1, 2)
fig.suptitle("Monopulse Angle Estimation")
axs[0].plot(snr_db_list, error_mean_list)
axs[0].set_title("Mean Angle Error")
axs[1].plot(snr_db_list, error_std_list)
axs[1].set_title(r"$\sigma$ Angle Error")

for ax in axs:
    ax.grid()
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Angle [Deg]")

plt.tight_layout()
plt.show()
