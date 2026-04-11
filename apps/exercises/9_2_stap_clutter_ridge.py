#!/usr/bin/env python
"""STAP clutter ridge and SINR loss.

Visualises *why* airborne clutter needs joint space-time processing, using the
theoretical clutter-plus-noise covariance (no data, no estimation noise):

1. Angle-Doppler clutter ridge -- the Capon (MVDR) power spectrum
   ``P(theta, fd) = 1 / (v^H R^-1 v)`` shows clutter concentrated on a ridge
   that couples angle and Doppler.  Platform motion makes a scatterer at
   azimuth ``theta`` appear at range-rate ``-v_platform * sin(theta)``, so the
   ridge is a curve, not a single Doppler line -- no fixed Doppler notch can
   remove it.

2. SINR loss vs Doppler -- ``L(fd) = (s^H R^-1 s) / (s^H s)`` at the broadside
   look direction.  It is ~0 dB (no loss) away from the ridge and drops into a
   deep notch where the target Doppler coincides with mainbeam clutter.  The
   width of that notch sets the minimum detectable velocity (MDV): targets
   slower than the notch edge are lost in clutter even with STAP.

The notch in (2) is exactly the Doppler slice of the ridge in (1) at the look
angle.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

from rad_lab import stap
from rad_lab import constants as c
from rad_lab.pulse_doppler_radar import Radar

# -- Radar / array (same platform as the STAP demonstration) --
bw = 1e6
radar = Radar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (20 / 10),
    rx_gain=10 ** (20 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (5 / 10),
    total_losses=10 ** (3 / 10),
    prf=10e3,
    dwell_time=1.6e-3,  # 16 pulses
)

n_elements = 8
el_pos = np.arange(n_elements) * 0.5 - (n_elements - 1) * 0.25
n_pulses = radar.n_pulses
platform_velocity = 100  # m/s
cnr = 40  # aggregate clutter-to-noise power ratio (~16 dB)

print("## STAP Clutter Ridge / SINR Loss ##")
print(f"  Array {n_elements} elements, {n_pulses} pulses -> NM = {n_elements * n_pulses} DoF")
print(f"  Platform velocity {platform_velocity} m/s, CNR {10 * np.log10(cnr):.0f} dB")

# -- Theoretical clutter-plus-noise covariance --
R = stap.clutter_plus_noise_covariance(el_pos, n_pulses, radar, platform_velocity, cnr=cnr)
R_inv = np.linalg.inv(R)

# -- (1) Angle-Doppler Capon spectrum --
angles = np.linspace(-90, 90, 181)
f_axis = fft.fftshift(fft.fftfreq(4 * n_pulses, 1 / radar.prf))  # oversampled Doppler
rdot_axis = -c.C * f_axis / (2 * radar.fcar)

capon = np.zeros((len(angles), len(f_axis)))
for i, ang in enumerate(angles):
    for j, fd in enumerate(f_axis):
        v = stap.space_time_steering_vector(el_pos, n_pulses, ang, fd, radar.prf)
        capon[i, j] = 1.0 / np.real(np.vdot(v, R_inv @ v))

capon_db = 10 * np.log10(capon / capon.max())

# -- (2) SINR loss vs Doppler at broadside --
sinr_db = np.array([
    10
    * np.log10(
        stap.sinr_loss(R, stap.space_time_steering_vector(el_pos, n_pulses, 0.0, fd, radar.prf))
    )
    for fd in f_axis
])

# The clutter ridge in range-rate: a broadside-looking platform sees a
# scatterer at azimuth theta with apparent range-rate -v_platform * sin(theta).
ridge_rdot = -platform_velocity * np.sin(np.deg2rad(angles))

# -- Plot --
fig, (ax_ridge, ax_sinr) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Clutter ridge and SINR loss (theoretical covariance)")

mesh = ax_ridge.pcolormesh(rdot_axis, angles, capon_db, shading="auto", vmin=-40, vmax=0)
ax_ridge.plot(ridge_rdot, angles, "r--", lw=1, label="ridge  $-v_p\\,\\sin\\theta$")
ax_ridge.set_title("Angle-Doppler Capon spectrum")
ax_ridge.set_xlabel("Range Rate [m/s]")
ax_ridge.set_ylabel("Angle [deg]")
ax_ridge.set_xlim(rdot_axis.min(), rdot_axis.max())  # ridge aliases past this
ax_ridge.legend(loc="upper right")
cbar = fig.colorbar(mesh, ax=ax_ridge)
cbar.set_label("Normalised power [dB]")

ax_sinr.plot(rdot_axis, sinr_db, ".-")
ax_sinr.axhline(-3, color="gray", ls=":", label="-3 dB")
ax_sinr.set_ylim(-30, 2)
ax_sinr.set_title("SINR loss at broadside look ($\\theta = 0$)")
ax_sinr.set_xlabel("Range Rate [m/s]")
ax_sinr.set_ylabel("SINR loss [dB]")
ax_sinr.legend(loc="lower right")
ax_sinr.grid(True, alpha=0.3)

fig.tight_layout()

# -- Report the minimum detectable velocity (clutter-notch half-width) --
# MDV: the smallest |range-rate| whose SINR loss has recovered to within 3 dB.
recovered = np.abs(rdot_axis[sinr_db > -3])
if recovered.size:
    print(
        f"  Minimum detectable velocity (SINR loss recovers to -3 dB): ~{recovered.min():.1f} m/s"
    )

plt.show()
