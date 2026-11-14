#!/usr/bin/env python
"""MTI blind speeds: the canceller removes targets too.

Three targets at different range rates are processed with and without a
2-pulse canceller.  The change in each target's SNR follows the canceller's
response at that target's Doppler.

Notes:
  - The 290 m/s target is near the first blind speed (~300 m/s).  Its Doppler
    aliases to almost zero, so the canceller treats it like clutter.
  - The 120 m/s target sits near the peak of the response and gains SNR.
  - Predicted change = |H(f)|^2 / noise_gain * (N - order) / N: the canceller's
    gain at the target's Doppler relative to its noise gain, and the loss of
    the dropped pulses from coherent integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm, mti, Radar, Target, Return, lfm_waveform
from rad_lab import pulse_doppler_radar as pdr

bw = 10e6  # waveform bandwidth [Hz]
order = 2  # MTI canceller order

radar = Radar(
    fcar=10e9,  # carrier frequency [Hz]
    tx_power=1e3,  # transmit power [W]
    tx_gain=10 ** (30 / 10),  # transmit antenna gain [linear], 30 dB
    rx_gain=10 ** (30 / 10),  # receive antenna gain [linear], 30 dB
    op_temp=290,  # operating temperature [K]
    sample_rate=2 * bw,  # Nyquist sampling of the waveform bandwidth
    noise_factor=10 ** (8 / 10),  # receiver noise factor [linear], 8 dB
    total_losses=10 ** (8 / 10),  # total system losses [linear], 8 dB
    prf=20e3,  # pulse repetition frequency [Hz]
    dwell_time=5e-3,  # coherent processing interval [s], 100 pulses
)

waveform = lfm_waveform(bw, T=1.0e-6, chirp_up_down=1)  # 1 us up-chirp

targets = [
    Target(range=2e3, range_rate=40, rcs=1),
    Target(range=3e3, range_rate=120, rcs=1),
    Target(range=4e3, range_rate=290, rcs=1),  # near the first blind speed
]
return_list = [Return(target=t) for t in targets]

print(f"first blind speed: {mti.blind_speeds(radar.prf, radar.fcar, n=1)[0]:.1f} m/s")

snr_db = {}
for mti_order in [0, order]:
    rdot_axis, r_axis, datacube = rdm.gen(
        radar, waveform, return_list, plot=False, mti_order=mti_order
    )
    snr_dc = rdm.to_snr(datacube, radar, waveform, mti_order=mti_order)

    snr_db[mti_order] = []
    for t in targets:
        # Range rate of each target after Doppler aliasing into [-prf/2, prf/2]
        rdot_alias = pdr.range_rate_aliased_prf_f0(t.range_rate, radar.prf, radar.fcar)
        i_r = np.argmin(abs(r_axis - t.range))
        i_v = np.argmin(abs(rdot_axis - rdot_alias))
        snr_db[mti_order].append(20 * np.log10(abs(snr_dc[i_r, i_v])))

    title = "no MTI" if mti_order == 0 else f"order-{mti_order} MTI"
    rdm.plot_rdm_snr(rdot_axis, r_axis, snr_dc, f"Three targets, {title}", cbar_min=0)

print(f"SNR change from order-{order} MTI (measured vs. predicted):")
for t, before, after in zip(targets, snr_db[0], snr_db[order]):
    doppler = pdr.frequency_delta_doppler(t.range_rate, radar.fcar)
    response = mti.frequency_response(order, radar.prf, np.array([doppler]))[0]
    pulse_ratio = (radar.n_pulses - order) / radar.n_pulses
    predicted = 10 * np.log10(response / mti.noise_gain(order) * pulse_ratio)
    print(f"  {t.range_rate:5.0f} m/s: {after - before:6.1f} dB vs {predicted:6.1f} dB")

plt.show()
