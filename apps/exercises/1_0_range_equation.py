#!/usr/bin/env python
"""Radar range equation exercises.

Problem 1: Plot SNR vs range for a BPSK waveform across different transmit
           powers and target RCS values.
Problem 2: Plot SNR vs range using the duty-factor form of the range equation
           across different CPI lengths and duty factors.
Problem 3: Compute maximum detection range as a 2D function of (Tx power, RCS)
           and (CPI time, RCS), visualized as heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab.constants import PI, C
import rad_lab.range_equation as re


## Problem 1: BPSK SNR vs range #############################################
# Explore how transmit power and target RCS affect detection range.

# -- Radar parameters --
antenna_diameter = 1.5 * (0.0254 * 12)  # 1.5 feet -> meters
fc = 10e9  # carrier frequency [Hz]
B = 10e6  # waveform bandwidth [Hz]
Ncode = 13  # number of chips in the Barker code
L = 10 ** (8 / 10)  # total system losses [linear], 8 dB
F = 10 ** (6 / 10)  # receiver noise factor [linear], 6 dB
n_p = 256  # number of pulses coherently integrated

# -- Sweep variables --
Pt_ar = [1e3, 5e3, 10e3]  # transmit power values [W]
sigma_db_ar = [0, 10, 20]  # target RCS values [dBsm]
sigma_ar = [10 ** (x / 10) for x in sigma_db_ar]  # convert RCS to linear [m^2]
R_ar = np.arange(1e3, 30.1e3, 100)  # range axis [m]
SNR_thresh_db = 12  # detection threshold [dB]

# -- Derived quantities --
wavelength = C / fc
theta_3db = wavelength / antenna_diameter  # 3-dB beamwidth for a circular aperture
Gt = 4 * PI / (theta_3db) ** 2  # transmit gain (az and el beamwidths equal)
Gr = Gt  # assume receive gain = transmit gain
T = 290  # system noise temperature [K]

# -- Plot SNR vs range for each (Pt, RCS) combination --
print(f"Problem 1: BPSK SNR [dB] at {R_ar[-1] * 1e-3:.0f} km")
fig, ax = plt.subplots(1, len(Pt_ar), sharex="all", sharey="all")
fig.suptitle("BPSK SNR")
for index, Pt in enumerate(Pt_ar):
    for sigma_index, sigma in enumerate(sigma_ar):
        y = re.snr_range_eqn_bpsk_cp(Pt, Gt, Gr, sigma, wavelength, R_ar, B, F, L, T, n_p, Ncode)
        y = 10 * np.log10(y)  # convert to dB
        print(f"\tPt={Pt * 1e-3:4.1f} kW  RCS={sigma_db_ar[sigma_index]:2d} dBsm : {y[-1]:5.1f}")
        ax[index].plot(R_ar / 1e3, y, label=f"RCS={sigma_db_ar[sigma_index]}[dBsm]")
    # overlay the detection threshold line
    ax[index].plot(
        R_ar / 1e3,
        SNR_thresh_db * np.ones(R_ar.shape),
        "--k",
        label=f"{SNR_thresh_db} dB threshold",
    )
    ax[index].set_title(f"Pt={Pt * 1e-3:.1f}[kW]")
    ax[index].set_xlabel("target distance [km]")
    ax[index].set_ylabel("SNR dB")
    ax[index].grid()

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncols=len(labels), bbox_to_anchor=(0.5, -0.02))
plt.subplots_adjust(bottom=0.15)

## Problem 2: Duty-factor range equation ####################################
# Explore how CPI length and duty factor affect SNR vs range.

sigma = 10 ** (0 / 10)  # 0 dBsm target
Pt = 5e3  # transmit power [W]
Tcpi_ar = [2e-3, 5e-3, 10e-3]  # CPI durations [s]
duty_factor_ar = [0.01, 0.1, 0.2]  # duty factors: 1%, 10%, 20%

print(f"Problem 2: duty-factor SNR [dB] at {R_ar[-1] * 1e-3:.0f} km")
fig, ax = plt.subplots(1, len(Tcpi_ar), sharex="all", sharey="all")
fig.suptitle("CPI duty-factor SNR")
for index, Tcpi in enumerate(Tcpi_ar):
    for duty_factor in duty_factor_ar:
        y = re.snr_range_eqn_duty_factor_pulses(
            Pt, Gt, Gr, sigma, wavelength, R_ar, F, L, T, Tcpi, duty_factor
        )
        y = 10 * np.log10(y)  # convert to dB
        print(f"\tTcpi={Tcpi * 1e3:4.1f} ms  DF={duty_factor:4.2f} : {y[-1]:5.1f}")
        ax[index].plot(R_ar / 1e3, y, label=f"DF={duty_factor}")
    ax[index].plot(
        R_ar / 1e3,
        SNR_thresh_db * np.ones(R_ar.shape),
        "--k",
        label=f"{SNR_thresh_db} dB threshold",
    )
    ax[index].set_title(f"CPI={Tcpi * 1e3:.1f}[ms]")
    ax[index].set_xlabel("target distance [km]")
    ax[index].set_ylabel("SNR dB")
    ax[index].grid()

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncols=len(labels), bbox_to_anchor=(0.5, -0.02))
plt.subplots_adjust(bottom=0.15)

## Problem 3: Maximum detection range heatmaps ##############################
# Solve the range equation for the range at which SNR drops to the detection
# threshold — the farthest range at which the target is still detectable —
# and visualize how it depends on (transmit power, RCS) and (CPI time, RCS).

SNR_thresh_db = 15
SNR_thresh = 10 ** (SNR_thresh_db / 10)
duty_factor = 0.1  # 10%
Tcpi = 2e-3  # [s]

# -- Figure 1: max detection range vs (Tx power, RCS) --
Pt_ar = np.arange(500, 10.1e3, 100)
sigma_db_ar = np.arange(-5, 26, 1)
sigma_ar = [10 ** (x / 10) for x in sigma_db_ar]
max_det_range_sigmaPt = np.zeros((len(Pt_ar), len(sigma_ar)))

for i, Pt in enumerate(Pt_ar):
    for j, sigma in enumerate(sigma_ar):
        val = re.max_target_detection_range_dutyfactor_cp(
            Pt, Gt, Gr, sigma, wavelength, SNR_thresh, F, L, T, Tcpi, duty_factor
        )
        max_det_range_sigmaPt[i, j] = val

print("Problem 3: max detection range grids [km]")
print(
    f"\t(Pt, RCS) grid:   min={max_det_range_sigmaPt.min() * 1e-3:.1f} "
    f"max={max_det_range_sigmaPt.max() * 1e-3:.1f}"
)

plt.figure()
plt.title(f"Tcpi={Tcpi * 1e3}[ms] DF={duty_factor}  SNR_thresh={SNR_thresh_db}[dB]")
plt.pcolormesh(sigma_db_ar, Pt_ar * 1e-3, max_det_range_sigmaPt * 1e-3)
c = plt.colorbar()
c.set_label("maximum detection range [km]")
plt.xlabel("target RCS [dBsm]")
plt.ylabel("transmit power [kW]")

# -- Figure 2: max detection range vs (CPI time, RCS) --
Pt = 5e3  # [W]
Tcpi_ar = np.arange(1e-3, 50.2e-3, 200e-6)
max_det_range_sigmaTcpi = np.zeros((len(Tcpi_ar), len(sigma_ar)))

for i, Tcpi in enumerate(Tcpi_ar):
    for j, sigma in enumerate(sigma_ar):
        val = re.max_target_detection_range_dutyfactor_cp(
            Pt, Gt, Gr, sigma, wavelength, SNR_thresh, F, L, T, Tcpi, duty_factor
        )
        max_det_range_sigmaTcpi[i, j] = val

print(
    f"\t(Tcpi, RCS) grid: min={max_det_range_sigmaTcpi.min() * 1e-3:.1f} "
    f"max={max_det_range_sigmaTcpi.max() * 1e-3:.1f}"
)

plt.figure()
plt.title(f"Pt={Pt * 1e-3:.1f}kW  DF={duty_factor}  SNR_thresh={SNR_thresh_db}[dB]")
plt.pcolormesh(sigma_db_ar, Tcpi_ar * 1e3, max_det_range_sigmaTcpi * 1e-3)
c = plt.colorbar()
c.set_label("maximum detection range [km]")
plt.xlabel("target RCS [dBsm]")
plt.ylabel("CPI time [ms]")

plt.show()
