#!/usr/bin/env python
"""Skin return with SNR verification.

Generate an RDM for a single moving target using an LFM waveform, normalise
to SNR voltage ratio with :func:`rdm.to_snr`, and compare the measured peak
SNR in the RDM to the range-equation prediction printed by
:func:`check_expected_snr`.
"""

import matplotlib.pyplot as plt
from rad_lab import rdm, Radar, Target, Return, lfm_waveform
from rad_lab._rdm_extras import verify_snr

bw = 10e6  # waveform bandwidth [Hz]

# -- Radar system --
radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,  # Nyquist rate for the waveform
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=200e3,
    dwell_time=2e-3,
)

# -- LFM waveform: 1 us up-chirp --
waveform = lfm_waveform(bw, T=1.0e-6, chirp_up_down=1)

# -- Target at 3.5 km, closing at 1 km/s, 10 m^2 RCS --
return_list = [Return(target=Target(range=3.5e3, range_rate=1.0e3, rcs=10))]

# -- Generate the RDM, then view it in SNR --
rdot_axis, r_axis, datacube = rdm.gen(radar, waveform, return_list, plot=False)
snr_dc = rdm.to_snr(datacube, radar, waveform)
rdm.plot_rdm_snr(rdot_axis, r_axis, snr_dc, f"SNR RDM for {waveform.type}", cbar_min=0)
verify_snr(snr_dc, radar, return_list[0].target, waveform)

plt.show()
