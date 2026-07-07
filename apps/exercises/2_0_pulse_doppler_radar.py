#!/usr/bin/env python
"""Pulse-Doppler radar ambiguity exercises.

Problem 1: Unambiguous range vs PRF.
Problem 2: Unambiguous range rate vs PRF for several carrier frequencies.
Problem 3: Range aliasing — where a 15.5 km target appears at various PRFs.
Problem 4: Doppler and range-rate aliasing vs PRF.
Problem 5: Range-rate aliasing vs carrier frequency at fixed PRF.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import pulse_doppler_radar as pdr


plt.rcParams["text.usetex"] = True

## Problem 1: Unambiguous range vs PRF ######################################
# Higher PRF => shorter PRI => less time for echoes => shorter unambiguous range.
PRF_ar = np.arange(1e3, 200.5e3, 500)

print("Problem 1: unambiguous range")
for PRF in [1e3, 10e3, 100e3]:
    print(f"\tPRF={PRF * 1e-3:5.0f} kHz : {pdr.range_unambiguous(PRF) * 1e-3:.2f} km")

plt.figure()
plt.plot(PRF_ar * 1e-3, pdr.range_unambiguous(PRF_ar) * 1e-3)
plt.title("unambiguous range vs PRF")
plt.xlabel("PRF [kHz]")
plt.ylabel("range [km]")
plt.yscale("log")
plt.grid(which="both")


## Problem 2: Unambiguous range rate vs PRF ##################################
# The maximum unambiguous range rate depends on both PRF and carrier frequency.
# Higher carrier frequency means smaller wavelength, so Doppler aliasing
# happens at lower velocities.
f0_ar = np.array([1, 2, 3, 4, 12, 16, 35, 95]) * 1e9  # carrier frequencies [Hz]

plt.figure()
plt.title(r"unambiguous $\pm$ range rate vs PRF")
plt.xlabel("PRF [kHz]")
plt.ylabel(r"$\pm$ max range rate [km/s]")
for f0 in f0_ar:
    plt.plot(
        PRF_ar * 1e-3, pdr.range_rate_pm_unambiguous(PRF_ar, f0) * 1e-3, label=f"$f_0=${f0:.1e}"
    )
plt.yscale("log")
plt.legend()
plt.grid()


## Problem 3: Range aliasing ################################################
# A target at 15.5 km folds into shorter apparent ranges when
# PRF pushes the unambiguous range below the true range.
R_tgt = 15.5 * 1e3  # true target range [m]
PRF_ar = np.array([2, 4, 8, 16, 32, 50, 60, 64, 95, 100, 128, 150, 228]) * 1e3  # [Hz]

apparent_range_ar = []
range_max_ar = []

for PRF in PRF_ar:
    range_max = pdr.range_unambiguous(PRF)
    apparent_range_ar.append(R_tgt % range_max)  # modulo gives the aliased range
    range_max_ar.append(range_max)

apparent_range_ar = np.array(apparent_range_ar)
range_max_ar = np.array(range_max_ar)

print(f"Problem 3: apparent range of a {R_tgt * 1e-3:.1f} km target")
for PRF, apparent in zip(PRF_ar, apparent_range_ar):
    print(f"\tPRF={PRF * 1e-3:5.0f} kHz : {apparent * 1e-3:6.2f} km")

plt.figure()
plt.title(f"range aliasing vs PRF: target range = {R_tgt * 1e-3:.1f} [km]")
plt.plot(PRF_ar * 1e-3, apparent_range_ar * 1e-3, "o", label="apparent range")
plt.plot(PRF_ar * 1e-3, range_max_ar * 1e-3, "--r", label="unambiguous range")
plt.xlabel("PRF [kHz]")
plt.ylabel("range [km]")
plt.ylim((0, 20))
plt.legend()
plt.grid()


## Problem 4: Doppler and range-rate aliasing ################################
# Show how a target's Doppler frequency and range rate alias as PRF changes.
rangeRate_tgt = -750  # target range rate [m/s]
f0 = 10e9  # carrier frequency [Hz]
apparent_doppler_ar = []
apparent_rangeRate_ar = []

for PRF in PRF_ar:
    # compute true Doppler, then alias it into [0, PRF) band
    doppler_freq_tgt = pdr.frequency_delta_doppler(rangeRate_tgt, f0)
    apparent_doppler_ar.append(pdr.frequency_aliased(doppler_freq_tgt, PRF))
    apparent_rangeRate_ar.append(pdr.range_rate_aliased_prf_f0(rangeRate_tgt, PRF, f0))

apparent_doppler_ar = np.array(apparent_doppler_ar)
apparent_rangeRate_ar = np.array(apparent_rangeRate_ar)

print(f"Problem 4: apparent Doppler and range rate of a {rangeRate_tgt} m/s target")
for PRF, doppler, rangeRate in zip(PRF_ar, apparent_doppler_ar, apparent_rangeRate_ar):
    print(f"\tPRF={PRF * 1e-3:5.0f} kHz : {doppler * 1e-3:7.2f} kHz  {rangeRate:7.1f} m/s")

fig, ax = plt.subplots(1, 2)
fig.suptitle(f"target range rate = {rangeRate_tgt} [m/s]")
ax[0].set_title("Doppler frequency aliasing")
ax[0].plot(PRF_ar * 1e-3, apparent_doppler_ar * 1e-3, "-o", label="apparent $f_D$")
ax[0].plot(PRF_ar * 1e-3, PRF_ar / 2 * 1e-3, "--r", label="unambiguous $f_D$")
ax[0].set_xlabel("PRF [kHz]")
ax[0].set_ylabel("apparent Doppler freq [kHz]")
ax[0].legend()
ax[0].grid()
ax[1].set_title("range rate aliasing")
ax[1].plot(PRF_ar * 1e-3, apparent_rangeRate_ar, "-o", label=r"apparent $\dot{r}$")
ax[1].plot(
    PRF_ar * 1e-3, pdr.range_rate_pm_unambiguous(PRF_ar, f0), "--r", label=r"unambiguous $\dot{r}$"
)
ax[1].plot(PRF_ar * 1e-3, -pdr.range_rate_pm_unambiguous(PRF_ar, f0), "--r")
ax[1].set_xlabel("PRF [kHz]")
ax[1].set_ylabel("apparent range rate [m/s]")
ax[1].legend()
ax[1].grid()

## Problem 5: Range-rate aliasing vs carrier frequency #######################
# At fixed PRF, sweeping carrier frequency changes the unambiguous range-rate
# window. Higher carrier => narrower window => more aliasing.
rangeRate_tgt = -750  # [m/s]
PRF = 16e3  # [Hz]
f0_ar = np.array([1, 2, 4, 6, 8, 10, 12, 16, 18, 34, 36, 94]) * 1e9

apparent_rangeRate_ar = []
rangeRate_max_ar = []
for f0 in f0_ar:
    rangeRate_max = pdr.range_rate_pm_unambiguous(PRF, f0)
    rangeRate_max_ar.append(rangeRate_max)
    apparent_rangeRate_ar.append(pdr.range_rate_aliased_prf_f0(rangeRate_tgt, PRF, f0))

apparent_rangeRate_ar = np.array(apparent_rangeRate_ar)
rangeRate_max_ar = np.array(rangeRate_max_ar)

print(
    f"Problem 5: apparent range rate of a {rangeRate_tgt} m/s target at PRF={PRF * 1e-3:.0f} kHz"
)
for f0, rangeRate in zip(f0_ar, apparent_rangeRate_ar):
    print(f"\tf0={f0 * 1e-9:4.0f} GHz : {rangeRate:7.1f} m/s")

plt.figure()
plt.title(
    rf"range rate vs freq: $\dot{{r}}_{{target}}$ = {rangeRate_tgt} [m/s], PRF={PRF * 1e-3:.1f}[kHz]"
)
plt.plot(f0_ar * 1e-9, apparent_rangeRate_ar, "o", label=r"apparent $\dot{r}$")
plt.plot(f0_ar * 1e-9, rangeRate_max_ar, "--r", label=r"unambiguous $\dot{r}$")
plt.plot(f0_ar * 1e-9, -rangeRate_max_ar, "--r")
plt.xlabel("operation frequency [GHz]")
plt.ylabel("apparent rangeRate [m/s]")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
