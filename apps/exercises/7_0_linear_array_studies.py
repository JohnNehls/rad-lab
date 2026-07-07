#!/usr/bin/env python
"""Uniform linear array (ULA) studies.

Study 1: Constant array length, varying element spacing.
  Shows that as long as dx <= lambda/2 (no grating lobes), the array factor
  shape is determined by the total array length — adding more elements
  only raises the array-factor peak (more coherent element sum), not the
  beamwidth.  Length here is N*dx (each element occupies a dx-wide cell):
  the beamwidth of a uniform array scales as 0.886/(N*dx) radians, so equal
  N*dx means equal beamwidth even though the tip-to-tip span (N-1)*dx
  differs slightly.

Study 2: Constant element spacing (lambda/2), varying number of elements.
  Demonstrates the Fourier-transform duality: a longer array produces a
  narrower mainlobe (higher angular resolution), analogous to how a longer
  time signal produces a narrower spectral peak.

Background: each array element receives the signal with a phase shift
  proportional to d*sin(theta)/lambda, where d is the element spacing.
"""

import matplotlib.pyplot as plt
import rad_lab.uniform_linear_arrays as ula


plt.rcParams["text.usetex"] = True


def print_pattern_summary(label, theta, gain_db):
    """Print the array-factor peak and 3-dB beamwidth of a pattern."""
    mainlobe = theta[gain_db >= gain_db.max() - 3]  # only the mainlobe is within 3 dB
    beamwidth = mainlobe[-1] - mainlobe[0]
    # 20*log10|AF| peaks at 20*log10(N) — the voltage array factor in dB,
    # not dBi (true directivity is ~10*log10(N) dBi).
    print(f"\t{label}: AF peak = {gain_db.max():5.2f} dB, 3-dB beamwidth = {beamwidth:.2f} deg")


## Study 1: Constant array length L = N*dx = 5*lambda, varying element density
# All three cases have the same N*dx, so the mainlobe width is the same and
# only the array-factor peak grows with N.
plt.figure()
plt.title(r"Unweighted Array Factor with Constant Length, L = N dx = 5 $\lambda$")
theta2, gain2 = ula.linear_antenna_gain_N_db(10, 1 / 2, plot=False)  # 10 el, dx=lambda/2
theta4, gain4 = ula.linear_antenna_gain_N_db(20, 1 / 4, plot=False)  # 20 el, dx=lambda/4
theta8, gain8 = ula.linear_antenna_gain_N_db(40, 1 / 8, plot=False)  # 40 el, dx=lambda/8
plt.plot(theta2, gain2, "-b", label=r"dx = $\lambda/2$, 10 elements")
plt.plot(theta4, gain4, "-.r", label=r"dx = $\lambda/4$, 20 elements")
plt.plot(theta8, gain8, "--k", label=r"dx = $\lambda/8$, 40 elements")
print("Study 1: constant length N*dx = 5 lambda (same beamwidth, AF peak grows with N)")
print_pattern_summary("dx=lambda/2, 10 el", theta2, gain2)
print_pattern_summary("dx=lambda/4, 20 el", theta4, gain4)
print_pattern_summary("dx=lambda/8, 40 el", theta8, gain8)
plt.xlabel(r"Angle $\theta$ [deg]")
plt.ylabel("Array factor [dB]")
plt.ylim((-60, 40))
plt.legend()
plt.grid()
plt.tight_layout()


## Study 2: Constant spacing dx = lambda/2, varying number of elements ######
# More elements => longer aperture => narrower beam.
plt.figure()
plt.title(r"Unweighted Array Factor with Constant Element Spacing, dx = $\lambda/2$")
# Lengths quoted as L = N*dx, the beamwidth-setting length (see Study 1).
theta2, gain2 = ula.linear_antenna_gain_N_db(4, 1 / 2, plot=False)  # L = 2*lambda
theta4, gain4 = ula.linear_antenna_gain_N_db(8, 1 / 2, plot=False)  # L = 4*lambda
theta8, gain8 = ula.linear_antenna_gain_N_db(16, 1 / 2, plot=False)  # L = 8*lambda
plt.plot(theta2, gain2, "-b", label=r"L = $2\lambda$, 4 elements")
plt.plot(theta4, gain4, "-.r", label=r"L = $4\lambda$, 8 elements")
plt.plot(theta8, gain8, "--k", label=r"L = $8\lambda$, 16 elements")
print("Study 2: constant spacing dx = lambda/2 (longer aperture => narrower beam)")
print_pattern_summary("L=2 lambda,  4 el", theta2, gain2)
print_pattern_summary("L=4 lambda,  8 el", theta4, gain4)
print_pattern_summary("L=8 lambda, 16 el", theta8, gain8)
plt.xlabel(r"Angle $\theta$ [deg]")
plt.ylabel("Array factor [dB]")
plt.ylim((-60, 30))
plt.legend()
plt.grid()
plt.tight_layout()


plt.show()
