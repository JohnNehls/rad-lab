#!/usr/bin/env python
"""Stationary target RDM snapshot with SNR check.

Generate a range-Doppler map for a single stationary target (range_rate=0)
using an LFM waveform and verify the SNR matches the range equation prediction.

Notes:
  - The Matlab reference solution underestimates SNR because it omits the
    time-bandwidth product gain from pulse compression.
  - Range walk-off is negligible here since the target is stationary.
"""

import matplotlib.pyplot as plt
from rad_lab import rdm
from rad_lab._rdm_extras import verify_snr
from rad_lab.pulse_doppler_radar import Radar
from rad_lab.waveform import lfm_waveform
from rad_lab.returns import Target, Return


bw = 10e6  # waveform bandwidth [Hz]

# -- Define the radar system --
radar = Radar(
    fcar=10e9,  # carrier frequency [Hz]
    tx_power=1e3,  # transmit power [W]
    tx_gain=10 ** (30 / 10),  # transmit antenna gain [linear], 30 dB
    rx_gain=10 ** (30 / 10),  # receive antenna gain [linear], 30 dB
    op_temp=290,  # operating temperature [K]
    sample_rate=2 * bw,  # Nyquist sampling of the waveform bandwidth
    noise_factor=10 ** (8 / 10),  # receiver noise factor [linear], 8 dB
    total_losses=10 ** (8 / 10),  # total system losses [linear], 8 dB
    prf=200e3,  # pulse repetition frequency [Hz]
    dwell_time=2e-3,  # coherent processing interval [s]
)

# -- Define the LFM waveform --
waveform = lfm_waveform(bw, T=1.0e-6, chirp_up_down=1)  # 1 us up-chirp

# -- Define the target: stationary at 3.5 km, 10 dBsm RCS --
return_list = [Return(target=Target(range=3.5e3, range_rate=0.0e3, rcs=10))]

# -- Generate the RDM, view in SNR, and verify against the range equation --
rdot_axis, r_axis, datacube = rdm.gen(radar, waveform, return_list, plot=False)
snr_dc = rdm.to_snr(datacube, radar, waveform)
rdm.plot_rdm_snr(rdot_axis, r_axis, snr_dc, f"SNR RDM for {waveform.type}", cbar_min=0)
verify_snr(snr_dc, radar, return_list[0].target, waveform)

plt.show()
