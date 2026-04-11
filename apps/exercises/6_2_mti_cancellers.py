#!/usr/bin/env python
"""MTI canceller demonstration.

Generates a datacube with ground clutter and two moving targets, then compares:
1. Conventional Doppler processing (no MTI) — clutter dominates.
2. 2-pulse canceller + Doppler — single clutter null at DC.
3. 3-pulse canceller + Doppler — double null, broader clutter suppression.

Also plots the frequency response of each canceller to show the null depth
and blind-speed notch widths.

Key takeaways:
- Ground clutter from a stationary platform is concentrated at zero Doppler.
  Without MTI, it can mask nearby moving targets.
- A 2-pulse canceller places a single null at DC, suppressing stationary
  clutter but leaving a narrow blind zone around each PRF multiple.
- A 3-pulse canceller has a broader null, suppressing clutter with a
  wider spectral spread — at the cost of wider blind zones.
- Higher-order cancellers trade blind-zone width for clutter rejection depth.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

from rad_lab import mti
from rad_lab.pulse_doppler_radar import Radar
from rad_lab.waveform import lfm_waveform
from rad_lab.rf_datacube import data_cube, range_axis, matchfilter
from rad_lab.noise import unity_variance_complex_noise
from rad_lab._rdm_internals import add_skin, skin_snr_amplitude
from rad_lab.returns import Target

# -- Radar and waveform --
bw = 5e6
waveform = lfm_waveform(bw, T=1e-6, chirp_up_down=1)

radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (5 / 10),
    total_losses=10 ** (3 / 10),
    prf=10e3,
    dwell_time=6.4e-3,  # 64 pulses — enough for decent Doppler resolution
)

waveform.set_sample(radar.sample_rate)
np.random.seed(0)

# -- Build the datacube --
dc = data_cube(radar.sample_rate, radar.prf, radar.n_pulses)
r_ax = range_axis(radar.sample_rate, dc.shape[0])

# -- Add two moving targets --
# Target 1: 5 km, -300 m/s (strong, well away from zero Doppler)
# Target 2: 3 km, -50 m/s (weaker, close to clutter in Doppler)
targets = [
    Target(range=5e3, range_rate=-300, rcs=100),
    Target(range=3e3, range_rate=-50, rcs=10),
]

for tgt in targets:
    amp = skin_snr_amplitude(radar, tgt, waveform)
    add_skin(dc, waveform, tgt, radar, amp)

# -- Add ground clutter (zero-Doppler, all range bins) --
# CNR per range bin: 40 dB above noise floor
cnr_linear = 10 ** (40 / 10)
clutter_amp = np.sqrt(cnr_linear / radar.n_pulses)
clutter = np.outer(
    unity_variance_complex_noise(dc.shape[0]) * clutter_amp,
    np.ones(radar.n_pulses),
)
dc += clutter

# -- Add noise --
noise = unity_variance_complex_noise(dc.shape) / np.sqrt(radar.n_pulses)
dc += noise

# -- Match filter --
matchfilter(dc, waveform.pulse_sample, pedantic=False)


# -- Helper: Doppler-process a datacube and return RDM + axes --
def doppler_rdm(datacube, prf):
    """FFT along slow time, return (rdot_axis, magnitude_dB)."""
    n_r, n_p = datacube.shape
    f_axis = fft.fftshift(fft.fftfreq(n_p, 1 / prf))
    rdm = fft.fftshift(fft.fft(datacube, axis=1), axes=1)
    rdot_axis = -3e8 * f_axis / (2 * radar.fcar)
    mag = np.abs(rdm)
    mag[mag == 0] = np.finfo(float).tiny
    return rdot_axis, 20 * np.log10(mag)


# -- Process: no MTI, 2-pulse, 3-pulse --
cases = [
    ("No MTI", dc),
    ("2-pulse MTI", mti.apply(dc, mti.canceller_weights(2))),
    ("3-pulse MTI", mti.apply(dc, mti.canceller_weights(3))),
]

# -- Figure 1: Side-by-side RDMs --
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (label, filtered_dc) in zip(axes, cases):
    rdot, db = doppler_rdm(filtered_dc, radar.prf)
    db_max = db.max()
    mesh = ax.pcolormesh(rdot * 1e-3, r_ax * 1e-3, db)
    mesh.set_clim(db_max - 60, db_max)
    ax.set_xlabel("Range Rate [km/s]")
    ax.set_ylabel("Range [km]")
    ax.set_title(label)
    fig.colorbar(mesh, ax=ax, label="dB")

fig.suptitle("MTI Clutter Suppression Comparison")
fig.tight_layout()

# -- Figure 2: Canceller frequency responses --
mti.plot_frequency_response(
    [mti.canceller_weights(n) for n in [2, 3, 4]],
    labels=["2-pulse", "3-pulse", "4-pulse"],
    prf=radar.prf,
)

# -- Print summary --
print("=" * 50)
print("MTI Canceller Demonstration")
print("=" * 50)
print(f"PRF: {radar.prf / 1e3:.0f} kHz, Pulses: {radar.n_pulses}")
print("Clutter: 40 dB CNR, zero Doppler")
print("Target 1: 5 km, -300 m/s, 20 dBsm")
print("Target 2: 3 km,  -50 m/s, 10 dBsm")

for label, filtered_dc in cases:
    rdot, db = doppler_rdm(filtered_dc, radar.prf)
    print(f"\n{label}: peak = {db.max():.1f} dB, shape = {filtered_dc.shape}")

plt.show()
