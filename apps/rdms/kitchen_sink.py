#!/usr/bin/env python
"""Kitchen-sink example: demonstrates all major RDM generation options.

This script shows how to combine:
  - Different waveform types (uncoded, Barker, random-coded, LFM)
  - A passive skin return (no jammer)
  - A DRFM jammer return with range/Doppler offsets and a steering vector
  - Debug and plotting options for rdm.gen()

The last waveform assignment wins, so uncomment the one you want to try.
"""

import matplotlib.pyplot as plt
import numpy as np
from rad_lab import rdm, Radar, Target, EaPlatform, Return
from rad_lab import uncoded_waveform, barker_coded_waveform, random_coded_waveform, lfm_waveform

bw = 10e6  # waveform bandwidth [Hz]

# -- Radar --
radar = Radar(
    fcar=10e9,
    tx_power=5e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=200e3,
    dwell_time=2e-3,
)

# -- Waveform options (last assignment wins) --
waveform = uncoded_waveform(bw)
waveform = barker_coded_waveform(bw, nchips=13)
waveform = random_coded_waveform(bw, nchips=13)
waveform = lfm_waveform(bw, T=10 / 40e6, chirp_up_down=1)

# -- Return 1: simple skin return (no jammer) --
skin_return = Return(target=Target(range=7.1e3, range_rate=-1e3, rcs=9))

# -- Return 2: target with a DRFM jammer --
# sv (steering vector) applies a complex phase shift, simulating an array element.
# The EaPlatform adds VBM noise with range/Doppler offsets and a retransmission delay.
jammer_on_target = Return(
    target=Target(range=3.5e3, range_rate=0.5e3, rcs=10, sv=np.exp(1j * np.pi / 4)),
    platform=EaPlatform(
        tx_power=1,  # jammer power [W]
        tx_gain=10 ** (5 / 10),  # jammer gain [linear], 5 dB
        total_losses=10 ** (5 / 10),  # jammer losses [linear], 5 dB
        rdot_delta=3.0e3,  # VBM Doppler spread [m/s]
        rdot_offset=0.3e3,  # Doppler offset [m/s]
        range_offset=-0.2e3,  # range offset [m]
        delay=1.33e-6,  # retransmission delay [s]
    ),
)

return_list = [skin_return, jammer_on_target]

# ----- Generate the RDM ------

# -- Windowing options: uncomment one to try it --
# rdm.gen(radar, waveform, return_list)                                             # Chebyshev 60 dB (default)
# rdm.gen(radar, waveform, return_list, window_kwargs={"at": 80})                   # Chebyshev 80 dB
# rdm.gen(radar, waveform, return_list, window="taylor")                            # Taylor (default nbar/sll)
# rdm.gen(radar, waveform, return_list, window="taylor", window_kwargs={"nbar": 6, "sll": -40})  # Taylor tuned
# rdm.gen(radar, waveform, return_list, window="blackman-harris")                   # Blackman-Harris
# rdm.gen(radar, waveform, return_list, window="none")                              # rectangular (no window)

# seed=0: reproducible noise; debug=True: show intermediate steps
rdot_axis, r_axis, datacube = rdm.gen(radar, waveform, return_list, seed=0, plot=True, debug=True)

plt.show()
