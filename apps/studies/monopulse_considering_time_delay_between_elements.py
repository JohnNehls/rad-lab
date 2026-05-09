#!/usr/bin/env python
"""Study: does the inter-element time delay matter for monopulse?

In a phased array, each element receives the signal at a slightly different
time due to the wavefront arrival angle. The standard monopulse model only
applies the carrier-frequency phase shift (exp(j*2*pi*d*sin(theta)/lambda_c)),
ignoring the baseband time delay.

This study compares two approaches:
  1. Phase-shift only (standard): multiply each element's datacube by the
     steering vector phase at the carrier frequency.
  2. Phase-shift + time delay: also apply the true time delay to the baseband
     signal, which shifts the waveform samples.

Findings:
  - When fcar >> bw, the time delay is negligible (lambda/2 at the carrier
    is lambda/100 at IF frequencies — very little phase change).
  - Regardless of fcar, the monopulse angle estimate is barely affected.
  - The study uses fcar=1 GHz with bw=200 MHz (ratio=5) to stress-test this.
"""

import matplotlib.pyplot as plt
import numpy as np
from rad_lab import constants as c
import rad_lab.uniform_linear_arrays as ula
from rad_lab.pulse_doppler_radar import Radar
from rad_lab.waveform import uncoded_waveform, barker_coded_waveform, lfm_waveform
from rad_lab.rdm import plot_rdm
import rad_lab.monopulse as mp
from rad_lab.rf_datacube import number_range_bins, range_axis, data_cube
from rad_lab.rf_datacube import matchfilter, doppler_process
from rad_lab.range_equation import noise_power
from rad_lab._rdm_internals import create_window, add_returns
from rad_lab.returns import Target, Return


# -- Use a low carrier-to-bandwidth ratio to make the time delay significant --
bw = 200e6  # waveform bandwidth [Hz]

radar = Radar(
    fcar=1e9,  # low carrier to stress-test the approximation
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=50e3,
    dwell_time=0.5e-3,
)

return_list = [Return(target=Target(range=2.4e3, range_rate=0.2e3, rcs=10))]

# -- Choose a waveform (last assignment wins) --
waveform = uncoded_waveform(bw)
waveform = barker_coded_waveform(bw, nchips=13)
waveform = lfm_waveform(3 * bw, T=10 / 40e6, chirp_up_down=1)

# -- Array geometry --
tgt_angle = 2  # target angle [deg]
dx = 1 / 2  # element separation [carrier wavelengths]
array_pos = np.array([-dx / 2, dx / 2])  # element positions [wavelengths]

## Step 1: Initialize waveform samples at the radar's sample rate #############
waveform.set_sample(radar.sample_rate)

## Step 2: Create datacube and populate with target returns ###################
r_axis = range_axis(radar.sample_rate, number_range_bins(radar.sample_rate, radar.prf))
signal_dc = data_cube(radar.sample_rate, radar.prf, radar.n_pulses)

# Compute noise voltage for scaling
rxVolt_noise = np.sqrt(
    c.RADAR_LOAD * noise_power(radar.sample_rate, radar.noise_factor, radar.op_temp)
)
noise_dc = np.random.uniform(low=-1, high=1, size=signal_dc.shape) * rxVolt_noise
add_returns(signal_dc, waveform, return_list, radar)

## Step 3: Apply array effects — compare phase-only vs phase+timeshift ########
assert abs(tgt_angle) < 90
signal_dc_ula_list = []  # phase-shift only
signal_dc_ula_list_timeshift = []  # phase-shift + time delay

for pos in array_pos:
    # Method 1: carrier phase shift only (standard approach)
    signal_dc_sv = signal_dc.copy() * ula.steering_vector(pos, tgt_angle)
    signal_dc_ula_list.append(signal_dc_sv)

    # Method 2: also apply the baseband time shift from element position
    position_meters = c.C / radar.fcar * pos  # convert wavelengths to meters
    tmp_signal = signal_dc_sv.T.flatten()  # flatten datacube to a 1D signal
    shifted_signal = ula.apply_timeshift_due_to_element_position(
        tmp_signal, radar.sample_rate, position_meters, tgt_angle
    )
    signal_dc_shift = shifted_signal.reshape(tuple(reversed(signal_dc_sv.shape))).T
    signal_dc_ula_list_timeshift.append(signal_dc_shift)

# Plot imaginary parts (real parts are identical for symmetric arrays)
fun = np.imag
plt.figure()
plt.title("Imaginary component of signal")
plt.plot(fun(signal_dc_ula_list[0].T.flatten()), label="neg position")
plt.plot(fun(signal_dc_ula_list[1].T.flatten()), label="pos position")
plt.plot(fun(signal_dc_ula_list_timeshift[0].T.flatten()), "--", label="timeshift neg position")
plt.plot(fun(signal_dc_ula_list_timeshift[1].T.flatten()), "--", label="timeshift pos position")
plt.xlabel("sample")
plt.ylabel("amplitude")
plt.legend()

## Step 4: Process all datacubes (matched filter + Doppler FFT) ###############
rdm_list = signal_dc_ula_list + signal_dc_ula_list_timeshift

# Apply matched filter to each datacube
for dc in rdm_list:
    matchfilter(dc, waveform.pulse_sample, pedantic=True)

# Apply Chebyshev window in slow time to suppress Doppler sidelobes
chwin_norm_mat = create_window(signal_dc.shape, plot=False)
for dc in rdm_list:
    dc *= chwin_norm_mat

# Doppler process (FFT along slow time)
for dc in rdm_list:
    f_axis, r_axis = doppler_process(dc, radar.sample_rate)

# Convert frequency axis to range-rate axis
rdot_axis = -c.C * f_axis / (2 * radar.fcar)


## Step 5: Compare monopulse angle estimates ##################################
def monopulse(dx, dc_list, tgt_angle):
    f_measured_theta = mp.monopulse_angle_at_peak_deg(dc_list[0], dc_list[1], dx)
    f_measured_error = abs(f_measured_theta - tgt_angle)
    print(f"\t{f_measured_theta=} degrees")
    print(f"\t{f_measured_error=} degrees")


print("No-timeshift")
monopulse(dx, signal_dc_ula_list, tgt_angle)
print("timeshift")
monopulse(dx, signal_dc_ula_list_timeshift, tgt_angle)

plot_rdm(rdot_axis, r_axis, signal_dc_ula_list[0], "Noiseless RDM")

plt.show()
