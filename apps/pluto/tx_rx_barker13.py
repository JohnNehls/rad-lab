import numpy as np
import adi
import matplotlib.pyplot as plt
import rsp.waveform as wvf
import rsp.rf_datacube as rfd

plt.close("all")

sample_rate = 56e6  # Hz
center_freq = 915e6  # Hz
num_samps = 10000  # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate)  # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = -10  # Increase to increase tx power, valid range is -90 to 0 dB

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = (
    0.0  # dB, increase to increase the receive gain, but be careful not to saturate the ADC
)

####### Start: Transmit Pulse #######################################
# uncoded
tx_wvf = [1]
# Barker
tx_wvf = wvf.BARKER_DICT[13]

z_list = [0 for _ in range(1600)]
tx_wvf = np.repeat(tx_wvf, 16)  # 16 samples per symbol (rectangular
samples = np.append(tx_wvf, z_list)

# LFM
# _ , tx_wvf = wvf.lfm_pulse(sample_rate, 56e6, 200*1/sample_rate, 1, output_length_T=1, normalize=False)
# samples = np.append(tx_wvf, np.zeros(1600))

print(f"{len(tx_wvf)}")
print(f"{abs(tx_wvf).max()}")
print(f"{abs(tx_wvf).mean()}")

samples *= 2**14  # PlutoSDR expects samples to be between -2^14 and +2^14

####### END: Transmit Pulse #######################################
# Start the transmitter
sdr.tx_cyclic_buffer = True  # Enable cyclic buffers
sdr.tx(samples)  # start transmitting

# Clear buffer just to be safe
for i in range(0, 10):
    raw_data = sdr.rx()

# Receive samples
rx_samples = sdr.rx()
rx_samples = rx_samples / np.max(rx_samples)
print(rx_samples)

# Stop transmitting
sdr.tx_destroy_buffer()

# Calculate power spectral density (frequency domain version of signal)
psd = np.abs(np.fft.fftshift(np.fft.fft(rx_samples))) ** 2
psd_dB = 10 * np.log10(psd)
f = np.linspace(sample_rate / -2, sample_rate / 2, len(psd))

# Plot time domain
# N_display = 1000
N_display = len(rx_samples)
plt.figure(0)
plt.title("IQ Data")
plt.plot(np.real(rx_samples), label="real")
plt.plot(np.imag(rx_samples), "--", label="imag")
plt.plot(np.abs(rx_samples), "k--", label="abs")
plt.plot(np.real(tx_wvf), label="tx real")
plt.plot(np.imag(tx_wvf), "--", label="tx imag")
plt.xlabel("Time")
plt.legend()


# Plot freq domain
plt.figure(1)
plt.plot(f / 1e6, psd_dB)
plt.xlabel("Frequency [MHz]")
plt.ylabel("PSD")
plt.show()
plt.grid()

### PP
# rx_s = rx_samples*np.exp(-1j*np.mean(np.angle(rx_samples[0:20])))  # may need span a boundary and need adjusment
# current_mid = np.abs(rx_s).max() - np.abs(rx_s).min()
# rx_s = (np.abs(rx_s) + current_mid)*np.exp(1j*np.angle(rx_s))
# rx_s = rx_s/abs(rx_s).max()
# plt.figure(2)
# plt.title("PP IQ Data")
# plt.plot(np.real(rx_s[:N_display]), label='real')
# plt.plot(np.imag(rx_s[:N_display]),'--', label='imag')
# plt.plot(np.real(tx_wvf), label='tx real')
# plt.plot(np.imag(tx_wvf),'--', label='tx imag')
# # plt.plot(np.abs(rx_s[:N_display]),'k--', label='abs')
# plt.xlabel("Time")
# plt.legend()

# Plot time domain
plt.figure(3)
plt.title("MF data")
idx_shift, mf_data = rfd.matchfilter_with_waveform(rx_samples, tx_wvf)
# idx_shift, mf_data = rfd.matchfilter_with_waveform(rx_s, tx_wvf[::-1])
# plt.plot(idx_shift, np.real(mf_data), label="real")
# plt.plot(idx_shift, np.imag(mf_data),'--', label="real")
# # plt.plot(idx_shift, abs(mf_data),'--k', label="abs")
# plt.plot(np.real(mf_data), label="real")
# plt.plot(np.imag(mf_data),'--', label="real")
plt.plot(abs(mf_data), "-k", label="abs")
# plt.plot(abs(mf_data[int(len(tx_wvf)/2):]),'-k', label="abs")  # front of pulse
plt.xlabel("Time")
plt.legend()
