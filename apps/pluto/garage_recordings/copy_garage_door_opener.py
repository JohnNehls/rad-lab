#!/usr/bin/env python

import time
import numpy as np
import adi
import matplotlib.pyplot as plt

plt.close("all")

# Notes:
# waveform is ~0.8 seconds long

TRANSMIT = True  # False means record

sample_rate = 2e6  # Hz
center_freq = 315e6  # Hz
num_samps = int(4e6)  # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

if TRANSMIT:
    sdr.tx_rf_bandwidth = int(sample_rate)  # filter cutoff, just set it to the same as sample rate
    sdr.tx_lo = int(center_freq)
    sdr.tx_hardwaregain_chan0 = 0  # Increase to increase tx power, valid range is -90 to 0 dB

    # load
    g_arr = np.load("sr_2MHz_lo_315MHz.npy")
    g_arr = g_arr / np.abs(g_arr).max()
    g_arr *= 2**14  # PlutoSDR expects samples to be between -2^14 and +2^14+-5026+3

    # sdr.tx_cyclic_buffer = True # Enabccle cyclic buffers
    sdr.tx(g_arr)  # start transmitting

    plt.figure(0)
    plt.title("TX IQ Data")
    t_axis = np.arange(len(g_arr)) / sample_rate
    plt.plot(t_axis, np.real(g_arr), label="real")
    plt.plot(t_axis, np.imag(g_arr), "--", label="imag")
    plt.xlabel("Time")
    plt.legend()

else:
    # elif True:
    # Config Rx
    sdr.rx_lo = int(center_freq)
    sdr.rx_rf_bandwidth = int(sample_rate)
    sdr.rx_buffer_size = num_samps
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = (
        0.0  # dB, increase to increase the receive gain, but be careful not to saturate the ADC
    )

    #### Receive samples
    # Clear buffer just to be safe
    for _ in range(2):
        raw_data = sdr.rx()

    print("PRESS!")
    # time.sleep(3.5) # 3 MHz sampeRate
    time.sleep(5)  # 2 MHz sampeRatex
    # time.sleep(1) # 0.5 MHz sampeRate
    print("start")
    rx_samples = sdr.rx()
    print("end")
    # Stop transmitting
    sdr.tx_destroy_buffer()

    # rx_samples = rx_samples/np.abs(rx_samples).max()
    print(rx_samples)

    #### Plot time domain
    # N_display = 1000
    N_display = len(rx_samples)
    # N_display = int(3e6)
    plt.figure(0)
    plt.title("IQ Data")
    t_axis = np.arange(len(rx_samples)) / sample_rate
    plt.plot(t_axis, np.real(rx_samples), label="real")
    plt.plot(t_axis, np.imag(rx_samples), "--", label="imag")
    # plt.plot(t_axis[::10], np.abs(rx_samples[::10]),'k--', label='abs')
    # plt.plot(np.abs(rx_samples[::1]),'k--', label='abs')
    plt.xlabel("Time")
    plt.legend()

    if False:
        # Calculate power spectral density (frequency domain version of signal)
        psd = np.abs(np.fft.fftshift(np.fft.fft(rx_samples))) ** 2
        psd_dB = 10 * np.log10(psd)
        f = np.linspace(sample_rate / -2, sample_rate / 2, len(psd))
        # Plot freq domain
        plt.figure(1)
        plt.plot(f / 1e6, psd_dB)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("PSD")
        plt.show()
        plt.grid()

    # from above plot, find the indices captureing the sequence
    # si = int(2.3e6)
    # ei = int(3.9e6)
    # plt.figure(0)
    # plt.title("IQ Data")
    # # plt.plot(np.real(rx_samples), label='real')
    # # plt.plot(np.imag(rx_samples),'--', label='imag')
    # plt.plot(np.abs(rx_samples[si:ei]),'k--', label='abs')
    # plt.xlabel("Time")
    # plt.legend()

    # # LAST STEP np.save out the array
    # np.save("sr_2MHz_lo_315MHz.npy", rx_samples[si:ei])
