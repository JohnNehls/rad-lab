#!/usr/bin/env python
"""MTI on a cluttered range-Doppler map.

Apply 0-, 1-, and 2-pulse cancellers to the same clutter scene as exercise
08_0 and compare the RDMs.  Then measure the improvement factor on a noiseless,
clutter-only datacube and compare it to `mti.improvement_factor`.

Notes:
  - MTI does not make the target easier to see here.  The 100-pulse Doppler
    FFT with a 60 dB window already separates the target from the clutter
    ridge, so the canceller lowers the ridge but attenuates the target and the
    background in its Doppler bin by the same amount.  Moving the target
    closer to the ridge only puts it in the canceller's notch.
  - MTI earns its keep when there is no Doppler FFT, as on a range profile
    or PPI display, where it is the only clutter filter (see section 10).  In
    real hardware it also relieves ADC dynamic range and phase-noise limits,
    which are not modelled here.
  - The canceller runs along slow time before the Doppler FFT, so it removes
    clutter from every Doppler bin, including the window sidelobes that leak
    clutter far from zero Doppler.
  - The canceller shapes the receiver noise too: after MTI the noise floor
    is darker near zero Doppler and brighter near +/- PRF/2, following
    |H(f)|^2.  `to_snr` normalises to the average noise gain.
  - Measured improvement factor = noise gain / clutter power gain.  Clutter
    power per pulse is compared before and after the canceller, so the
    pulses it drops do not bias the result.
"""

import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from rad_lab import rdm, mti, Radar, Target, Return, Clutter, lfm_waveform

bw = 10e6  # waveform bandwidth [Hz]

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

target = Target(range=4e3, range_rate=40, rcs=1)
clutter = Clutter(
    range_start=1e3,
    range_stop=6e3,
    gamma=10 ** (-20 / 10),  # -20 dB constant gamma
    grazing_angle=np.deg2rad(2),
    azimuth_beamwidth=np.deg2rad(2),
    range_rate_std=3.0,  # internal-motion spread [m/s]
)

# -- RDMs with MTI off, order 1, and order 2 --
for order in [0, 1, 2]:
    rdot_axis, r_axis, datacube = rdm.gen(
        radar, waveform, [Return(target=target), clutter], plot=False, mti_order=order
    )
    snr_dc = rdm.to_snr(datacube, radar, waveform, mti_order=order)

    i_r = np.argmin(abs(r_axis - target.range))
    i_v = np.argmin(abs(rdot_axis - target.range_rate))
    i_0 = np.argmin(abs(rdot_axis))
    print(f"MTI order {order}:")
    print(f"  target SNR       = {20 * np.log10(abs(snr_dc[i_r, i_v])):.1f} dB")
    print(f"  peak clutter CNR = {20 * np.log10(abs(snr_dc[:, i_0]).max()):.1f} dB")

    title = "no MTI" if order == 0 else f"order-{order} MTI"
    rdm.plot_rdm_snr(rdot_axis, r_axis, snr_dc, f"Clutter RDM, {title}", cbar_min=0)

# -- Measured vs. theoretical improvement factor (noiseless, clutter only) --
quiet_radar = dataclasses.replace(radar, op_temp=0)


def clutter_power_per_pulse(order: int) -> float:
    """Mean clutter power per range bin and pulse after an order-``order`` canceller."""
    _, r_axis, datacube = rdm.gen(
        quiet_radar, waveform, [clutter], plot=False, window="none", mti_order=order
    )
    band = (r_axis > clutter.range_start) & (r_axis < clutter.range_stop)
    # Unnormalised FFT: mean |X|^2 over Doppler bins = n_pulses * mean |x|^2 over pulses.
    return np.mean(np.abs(datacube[band]) ** 2) / datacube.shape[1]


power_in = clutter_power_per_pulse(0)
print("\nimprovement factor (measured vs. theory):")
for order in [1, 2]:
    clutter_gain = clutter_power_per_pulse(order) / power_in
    measured = mti.noise_gain(order) / clutter_gain
    theory = mti.improvement_factor(order, radar.prf, radar.fcar, clutter.range_rate_std)
    print(f"  order {order}: {10 * np.log10(measured):.1f} dB vs {10 * np.log10(theory):.1f} dB")

plt.show()
