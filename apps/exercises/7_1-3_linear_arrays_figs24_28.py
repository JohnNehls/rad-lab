#!/usr/bin/env python
"""Recreate textbook figures 24-28 for uniform linear arrays.

Figure 24: Effect of number of elements on the array factor (lambda/2 spacing).
Figure 25: Effect of element spacing on the array factor (10 elements).
           Note: spacing > lambda/2 introduces grating lobes.
Figure 27: Weighted array factors (Chebyshev and Taylor) vs unweighted.
           Weighting suppresses sidelobes at the cost of mainlobe width.
Figure 28: Beam steering to 15, 45, and -60 degrees.
           Steering broadens the beam and reduces peak gain at large angles.

Discrepancies with reference document:
  - Figures 24 and 25 appear to use 10*log10 instead of 20*log10 in the
    original document, causing a 2x scaling error.
  - Our figures 27 and 28 match the document correctly.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import rad_lab.uniform_linear_arrays as ula


plt.rcParams["text.usetex"] = True

## Figure 24: Unweighted, lambda/2 spacing, 10 vs 40 elements ##############
fig, axs = plt.subplots(1, 2)
fig.suptitle(r"Unweighted Array Factor for $\lambda/2$ spacing")

print("Figure 24: peak gain (20*log10(N) dBi; see 10 vs 20 log10 note above)")
theta, gain = ula.linear_antenna_gain_N_db(10, 1 / 2, plot=False)
axs[0].plot(theta, gain)
axs[0].set_title("10 elements")
print(f"\t10 elements: {gain.max():.2f} dBi")

theta, gain = ula.linear_antenna_gain_N_db(40, 1 / 2, plot=False)
axs[1].plot(theta, gain)
axs[1].set_title("40 elements")
print(f"\t40 elements: {gain.max():.2f} dBi")

for ax in axs:
    ax.grid()
    ax.set_ylim((-60, 40))
    ax.set_xlabel(r"Angle $\theta$ [deg]")
    ax.set_ylabel("Gain [dBi]")

plt.tight_layout()

## Figure 25: 10 elements, varying spacing ##################################
# lambda/4: oversampled, no grating lobes
# lambda/2: critical spacing, no grating lobes
# lambda:   undersampled, grating lobes appear at ±90 deg
fig, axs = plt.subplots(1, 3)
fig.suptitle("Unweighted Array Factor for 10 Elements")

# The normalized gain near +/-90 deg reveals grating lobes: ~0 dB when the
# spacing exceeds lambda/2, far below 0 dB otherwise.
print("Figure 25: normalized gain beyond +/-60 deg (0 dB indicates a grating lobe)")


def print_outer_gain(label, theta, gain):
    outer = (gain - gain.max())[abs(theta) > 60]
    print(f"\t{label} spacing: {outer.max():7.2f} dB")


theta, gain = ula.linear_antenna_gain_N_db(10, 1 / 4, plot=False)
axs[0].plot(theta, gain - gain.max())
axs[0].set_title(r"$\lambda/4$ spacing")
print_outer_gain("lambda/4", theta, gain)

theta, gain = ula.linear_antenna_gain_N_db(10, 1 / 2, plot=False)
axs[1].plot(theta, gain - gain.max())
axs[1].set_title(r"$\lambda/2$ spacing")
print_outer_gain("lambda/2", theta, gain)

theta, gain = ula.linear_antenna_gain_N_db(10, 1, plot=False)
axs[2].plot(theta, gain - gain.max())
axs[2].set_title(r"$\lambda$ spacing")
print_outer_gain("lambda  ", theta, gain)

for ax in axs:
    ax.grid()
    ax.set_ylim((-80, 10))
    ax.set_xlabel(r"Angle $\theta$ [deg]")
    ax.set_ylabel("Normalized Gain [dBi]")

plt.tight_layout()

## Figure 27: Weighted vs unweighted, 40 elements ##########################
Nel = 40
chebWindow = signal.windows.chebwin(Nel, 30)  # 30 dB Chebyshev sidelobe level
tayWindow = signal.windows.taylor(Nel, sll=35)  # 35 dB Taylor sidelobe level

fig, axs = plt.subplots(1, 2)
fig.suptitle(r"Array Factor with Different Weights: 40 elements, $\lambda/2$ spacing")

theta, gain = ula.linear_antenna_gain_N_db(Nel, 1 / 2, plot=False)
theta, gain_cheb = ula.linear_antenna_gain_N_db(Nel, 1 / 2, weight_vec=chebWindow, plot=False)
theta, gain_tay = ula.linear_antenna_gain_N_db(Nel, 1 / 2, weight_vec=tayWindow, plot=False)

# Peak sidelobe level: max normalized gain outside the mainlobe (|theta| > 5 deg)
print("Figure 27: peak sidelobe level")
for label, g in [("unweighted  ", gain), ("30 dB Cheb. ", gain_cheb), ("35 dB Taylor", gain_tay)]:
    sidelobes = (g - g.max())[abs(theta) > 5]
    print(f"\t{label}: {sidelobes.max():7.2f} dB")

# Left: full angular range; Right: zoomed into mainlobe
axs[0].set_xlim((-90, 90))
axs[1].set_xlim((-8, 8))

for ax in axs:
    ax.plot(theta, gain - gain.max(), label="unweighted")
    ax.plot(theta, gain_cheb - gain_cheb.max(), "-.r", label="30 dB Cheb.")
    ax.plot(theta, gain_tay - gain_tay.max(), "--k", label="35 dB Taylor")
    ax.set_xlabel(r"Angle $\theta$ [deg]")
    ax.set_ylabel("Normalized Gain [dBi]")
    ax.set_ylim((-60, 5))
    ax.grid()

axs[1].legend()
plt.tight_layout()

## Figure 28: Beam steering, 20 elements ###################################
# Applying a linear phase taper across elements steers the beam.
fig, axs = plt.subplots(1, 3)
fig.suptitle(r"Weighted Array Factor: $\lambda/2$ spacing, 20 elements")

Nel = 20
dx = 1 / 2
print("Figure 28: steered beam peaks")
for ax, angle in zip(axs, [15, 45, -60]):
    theta, gain = ula.linear_antenna_gain_N_db(Nel, dx, steer_angle=angle, plot=False)
    ax.plot(theta, gain - gain.max())
    ax.set_title(rf"Steered to {angle} deg")
    print(f"\tsteered to {angle:3d} deg: peak at {theta[np.argmax(gain)]:6.2f} deg")
    ax.grid()
    ax.set_ylim((-60, 5))
    ax.set_xlabel(r"Angle $\theta$ [deg]")
    ax.set_ylabel("Normalized Gain [dBi]")

plt.tight_layout()

plt.show()
