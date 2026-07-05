#!/usr/bin/env python
"""Compare VBM noise generation methods.

The VBM (Velocity-Bin Masking) jammer spreads energy across Doppler bins by
modulating the retransmitted signal's phase in slow time. There are several
ways to generate that phase modulation, listed here from simplest to most
physically accurate:

  1. random_phase: each pulse gets an independent random phase — produces
     flat Doppler noise but doesn't respect the prescribed bandwidth.
  2. uniform_bandwidth_phase: random phase filtered to a uniform spectral
     shape within the target bandwidth.
  3. gaussian_bandwidth_phase: random phase filtered to a Gaussian spectral
     shape — more realistic but doesn't match the prescribed width exactly.
  4. gaussian_bandwidth_phase_normalized: same as (3) but normalized.
  5. lfm_phase: slow-time LFM chirp — the only method that precisely matches
     the prescribed rdot_delta bandwidth.

Each method generates an RDM so you can visually compare the Doppler spread.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm, Radar, Target, EaPlatform, Return, uncoded_waveform
import rad_lab.vbm as vbm

bw = 5e6  # waveform bandwidth [Hz]

# -- Radar (high PRF and long dwell for fine Doppler resolution) --
radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=500e3,
    dwell_time=5e-3,
)

waveform = uncoded_waveform(bw)

# -- Map of VBM method names to their internal functions --
vbm_name_function_dict = {
    "random phase VBM": vbm._random_phase,
    "uniform bandwidth phase VBM": vbm._uniform_bandwidth_phase,
    "gaussian bandwidth phase VBM": vbm._gaussian_bandwidth_phase,
    "uniform bandwidth phase normalized VBM": vbm._gaussian_bandwidth_phase_normalized,
    "LFM phase VBM": vbm._lfm_phase,
}

# -- Generate an RDM for each VBM method --
# For each method, measure the Doppler extent within 20 dB of the peak at the
# jammer's range bin and compare it to the prescribed rdot_delta.
rdot_delta = 1.0e3  # prescribed Doppler spread [m/s]
print(f"measured VBM Doppler spread (prescribed rdot_delta = {rdot_delta:.0f} m/s)")
for name, func in vbm_name_function_dict.items():
    jammer_return = Return(
        target=Target(range=0.2e3, range_rate=0.0e3),
        platform=EaPlatform(
            tx_power=1.0e3,
            tx_gain=10 ** (5 / 10),
            total_losses=10 ** (3 / 10),
            rdot_delta=rdot_delta,
            rdot_offset=0.0e3,
            vbm_noise_function=func,
        ),
    )
    rdot_axis, r_axis, datacube = rdm.gen(radar, waveform, [jammer_return], debug=False)
    ax = plt.gca()
    ax.set_title(name)

    peak_r, _ = np.unravel_index(np.argmax(abs(datacube)), datacube.shape)
    row = abs(datacube[peak_r, :])
    masked = rdot_axis[row > row.max() / 10]
    print(f"\t{name}: {masked.max() - masked.min():7.1f} m/s")

print(
    f"Note:LFM phase VBM is the only method to match the perscribed {rdot_delta} [m/s] VBM width."
)
plt.show()
