#!/usr/bin/env python
"""Noisy cross-correlation (matched filter) exercises.

Demonstrate how a matched filter detects pulses buried in noise:
  Case 1: Single uncoded pulse at 20 dB SNR.
  Case 2: Three uncoded pulses at different SNRs (15, 30, 20 dB).
  Case 3: An LFM pulse and a Barker-13 BPSK pulse in the same noise,
          showing that each matched filter only responds to its own waveform.

Key takeaway: the matched-filter peak appears at the center of the pulse
(not the leading edge), and scales with the pulse's processing gain.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab.waveform_helpers import (
    matchfilter_with_waveform,
    add_waveform_at_index,
    plot_pulse_and_spectrum,
)
from rad_lab.noise import unity_variance_complex_noise
from rad_lab.waveform import uncoded_pulse, barker_coded_pulse, lfm_pulse

np.random.seed(0)  # reproducible noise realizations

print("##############################")
print("Problem 3: noisy xcorrelations")
print("##############################")

# -- Common waveform parameters --
sampleRate = 20e6  # sample rate [Hz]
BW = 4e6  # waveform bandwidth [Hz]

## Case 1: Single uncoded pulse in noise #####################################
print("## Case 1: add uncoded pulse ######")

# Generate unity-variance complex Gaussian noise (0 dB noise floor)
print("\tcreate noise")
noise_1 = unity_variance_complex_noise(1000)

# Verify the pulse bandwidth via its spectrum
print("verify uncoded pulse BW")
t_u, mag_u = uncoded_pulse(sampleRate, BW, normalize=True)
plot_pulse_and_spectrum(t_u, mag_u, "uncoded BW check", n_pad=1000)

# Create the pulse, scale it to the desired SNR, and embed it in noise
t_u, mag_u = uncoded_pulse(sampleRate, BW)
indx_1 = 200  # sample index where pulse is placed
SNR = 20  # desired SNR [dB] (noise floor is 0 dB)
mag_u_s = 10 ** (SNR / 20) * mag_u  # scale pulse voltage for target SNR

add_waveform_at_index(noise_1, mag_u_s, indx_1)  # add pulse into noise in-place
print("verify noise is at ~ 0dB")
print(f"\tnoise variance = {10 * np.log10(np.var(noise_1)):.2f} dB")

# Apply matched filter: correlate the noisy signal with the known pulse
m_index_shift, mf = matchfilter_with_waveform(noise_1, mag_u)
print(f"matched-filter peak at sample {np.argmax(abs(mf))} (pulse placed at index {indx_1})")

# Plot: noisy signal, original pulse, and matched-filter output
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(f"S3P3 case 1: uncoded pulse at index{indx_1}")
ax[0].plot(np.real(noise_1), "-o")
ax[0].set_title("noise")
ax[0].set_xlabel("sample")
ax[0].set_ylabel("real noise")
ax[1].plot(t_u, abs(mag_u_s), "-o")
ax[1].set_title("uncoded pulse")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("magnitude")
ax[2].plot(abs(mf), "-o")
ax[2].set_xlabel("sample")
ax[2].set_ylabel("matched magnitude")
ax[2].set_title("match filter on noise")
plt.tight_layout()
for a in ax:
    a.grid()

## Case 2: Three uncoded pulses at different SNRs ###########################
print("## Case 2: three uncoded pulse ####")
noise_2 = unity_variance_complex_noise(1000)

# Embed three pulses at different locations and SNR levels
indx_1 = 128
SNR = 15  # [dB]
_, mag_u = uncoded_pulse(sampleRate, BW)
mag_u_s = 10 ** (SNR / 20) * mag_u
add_waveform_at_index(noise_2, mag_u_s, indx_1)

indx_2 = 200
SNR = 30  # [dB] — strongest pulse
_, mag_u = uncoded_pulse(sampleRate, BW)
mag_u_s = 10 ** (SNR / 20) * mag_u
add_waveform_at_index(noise_2, mag_u_s, indx_2)

indx_3 = 950
SNR = 20  # [dB]
_, mag_u = uncoded_pulse(sampleRate, BW)
mag_u_s = 10 ** (SNR / 20) * mag_u
add_waveform_at_index(noise_2, mag_u_s, indx_3)

# Matched filter should show three distinct peaks at the pulse locations
_, mf = matchfilter_with_waveform(noise_2, mag_u)
print(f"strongest matched-filter peak at sample {np.argmax(abs(mf))} (30 dB pulse at {indx_2})")

fig, ax = plt.subplots(1, 4, figsize=(15, 4))
fig.suptitle(f"S3P3 case 2: uncoded pulses @ index: {indx_1}, {indx_2}, {indx_3} -- check SNR")
ax[0].plot(np.real(noise_2), "-o")
ax[0].set_title("noise")
ax[0].set_xlabel("sample")
ax[0].set_ylabel("real noise")
ax[1].plot(abs(mag_u_s), "-o")
ax[1].set_title("uncoded pulse")
ax[1].set_xlabel("sample")
ax[1].set_ylabel("magnitude")
ax[2].plot(abs(mf), "-o")
ax[2].set_xlabel("sample")
ax[2].set_ylabel("matched magnitude")
ax[2].set_title("match filter on noise")
ax[3].plot(10 * np.log(abs(mf)), "-o")
ax[3].set_xlabel("sample")
ax[3].set_ylabel("matched dB")
ax[3].set_title("match filter on noise")
plt.tight_layout()
for a in ax:
    a.grid()


## Case 3: LFM and BPSK pulses — waveform selectivity ######################
# Place an LFM chirp and a Barker-13 BPSK pulse in the same noise, then
# show that each matched filter only detects its own waveform type.
print("## Case 3: LFM and BPSK pulses ####")
noise_3 = unity_variance_complex_noise(1000)

print(f"noise variance = {10 * np.log10(np.var(noise_3)):.2f} dB")  # verify ~0 dB floor

# Embed an LFM pulse at sample 300
lfm_idx = 300
SNR = 20  # [dB]
T = 2e-6  # pulse duration [s]
chirpUpDown = 1  # up-chirp
_, mag_lfm = lfm_pulse(sampleRate, BW, T, chirpUpDown)
mag_lfm_s = 10 ** (SNR / 20) * mag_lfm
add_waveform_at_index(noise_3, mag_lfm_s, lfm_idx)

# Embed a Barker-13 BPSK pulse at sample 600
bpsk_idx = 600
SNR = 20  # [dB]
_, mag_b = barker_coded_pulse(sampleRate, BW, 13)
mag_b_s = 10 ** (SNR / 20) * mag_b
add_waveform_at_index(noise_3, mag_b_s, bpsk_idx)

# Each matched filter should peak only at its own waveform's location
_, mf_lfm = matchfilter_with_waveform(noise_3, mag_lfm)
_, mf_b = matchfilter_with_waveform(noise_3, mag_b)
print(f"lfm matched-filter peak at sample {np.argmax(abs(mf_lfm))} (lfm pulse at {lfm_idx})")
print(f"bpsk matched-filter peak at sample {np.argmax(abs(mf_b))} (bpsk pulse at {bpsk_idx})")

# Plot: combined signal, each pulse, and each matched-filter output
fig, ax = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("S3P3 case 3")
ax[0].plot(abs(noise_3))
ax[0].set_title("signal + noise")
ax[0].set_xlabel("sample")
ax[0].set_ylabel("magnitude")
ax[1].plot(abs(mag_lfm_s), "-o")
ax[1].set_title(f"lfm pulse, index={lfm_idx}")
ax[1].set_xlabel("sample")
ax[1].set_ylabel("magnitude")
ax[2].plot(abs(mag_b_s), "-o")
ax[2].set_title(f"bpsk pulse, index={bpsk_idx}")
ax[2].set_xlabel("sample")
ax[2].set_ylabel("magnitude")
# LFM matched filter — should peak at lfm_idx, suppress the BPSK pulse
ax[3].plot(abs(mf_lfm))
ax[3].set_title("lfm match")
ax[3].set_xlabel("sample")
ax[3].set_ylabel("magnitude")
# BPSK matched filter — should peak at bpsk_idx, suppress the LFM pulse
ax[4].plot(abs(mf_b))
ax[4].set_title("bpsk match")
ax[4].set_xlabel("sample")
ax[4].set_ylabel("magnitude")
plt.tight_layout()
for a in ax:
    a.grid()

plt.show()
