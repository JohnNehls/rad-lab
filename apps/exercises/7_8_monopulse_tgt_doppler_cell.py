#!/usr/bin/env python
"""Monopulse angle estimation on a range-Doppler map.

Demonstrate that monopulse processing can be applied after Doppler processing
(FFT along slow time). The workflow:

  1. Define a two-element array and compute steering vectors for the target angle.
  2. Generate a separate RDM for each array element (same noise seed so the
     noise realization is identical — only the signal phase differs).
  3. Apply monopulse at the peak cell of the RDMs to estimate the target angle.

This validates that the inter-element phase relationship is preserved through
matched filtering and Doppler processing.
"""

import matplotlib.pyplot as plt
import numpy as np
import rad_lab.uniform_linear_arrays as ula
import rad_lab.rdm as rdm
import rad_lab.monopulse as mp
from rad_lab.pulse_doppler_radar import Radar
from rad_lab.waveform import uncoded_waveform, barker_coded_waveform, lfm_waveform
from rad_lab.returns import Target, Return


bw = 10e6  # waveform bandwidth [Hz]

# -- Define the radar --
radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=50e3,
    dwell_time=2e-3,
)

# -- Choose a waveform (last one wins — try uncommenting others) --
waveform = uncoded_waveform(bw)  # high 1
waveform = barker_coded_waveform(bw, nchips=13)  # high 1
waveform = lfm_waveform(bw, T=10 / 40e6, chirp_up_down=1)  # high 2

# -- Two-element array and target --
tgt_angle = 5  # true target angle [deg]
dx = 1 / 2  # element separation [wavelengths]
array_pos = np.array([-dx / 2, dx / 2])
steer_vec = ula.steering_vector(array_pos, tgt_angle)

# -- Generate one RDM per array element --
# Each element sees the same target but with a different phase (steering vector).
dc_list = []
for sv in steer_vec:
    rseed = np.random.randint(1000)
    return_list = [Return(target=Target(range=2.4e3, range_rate=0.2e3, rcs=10, sv=sv))]
    rdot_axis, r_axis, datacube = rdm.gen(
        radar, waveform, return_list, debug=False, plot=False, seed=rseed
    )
    dc_list.append(datacube)

# -- Display each element's RDM --
for i in range(len(steer_vec)):
    rdm.plot_rdm(rdot_axis, r_axis, dc_list[i], f"{i=}", cbar_min=-100, volt_to_dbm=True)

# -- Apply monopulse at the peak RDM cell to estimate angle --
f_measured_theta = mp.monopulse_angle_at_peak_deg(dc_list[0], dc_list[1], dx)
f_measured_error = abs(f_measured_theta - tgt_angle)

print(f"{f_measured_theta=} degrees")
print(f"{f_measured_error=} degrees")

plt.show()
