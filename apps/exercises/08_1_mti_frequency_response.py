#!/usr/bin/env python
"""MTI canceller frequency response, blind speeds, and improvement factor.

Plot the power response of the 1-, 2-, and 3-pulse binomial cancellers against
target range rate, marking the blind speeds where the response is zero, then
plot the theoretical improvement factor against clutter spectral spread.

Notes:
  - Every canceller has nulls at zero Doppler (clutter) and at each multiple
    of the PRF.  Targets at those range rates, v_k = k * lambda * prf / 2, are
    cancelled along with the clutter.
  - Higher order widens the clutter notch (better clutter rejection) but also
    widens the band of poorly-detected slow targets.
  - The improvement factor falls steeply as the clutter spectrum widens;
    the higher-order cancellers lose their advantage fastest.
  - These are ideal values.  Real MTI improvement is capped at a few tens of
    dB by oscillator phase noise, timing jitter, and ADC dynamic range.
"""

import numpy as np
import matplotlib.pyplot as plt
from rad_lab import mti
from rad_lab import pulse_doppler_radar as pdr

fcar = 10e9  # carrier frequency [Hz]
prf = 20e3  # pulse repetition frequency [Hz]
orders = [1, 2, 3]

blind = mti.blind_speeds(prf, fcar, n=2)
print(f"blind speeds: {', '.join(f'{v:.1f}' for v in blind)} m/s")
for order in orders:
    print(f"order {order}: taps {mti.canceller_coefficients(order)}, ", end="")
    print(f"noise gain {mti.noise_gain(order):.0f}")

# -- Frequency response vs. range rate --
range_rates = np.linspace(-blind[1], blind[1], 2001)
freqs = pdr.frequency_delta_doppler(range_rates, fcar)  # |H|^2 is symmetric in Doppler

fig, ax = plt.subplots()
for order in orders:
    response = mti.frequency_response(order, prf, freqs)
    with np.errstate(divide="ignore"):
        ax.plot(range_rates, 10 * np.log10(response), label=f"order {order}")
for v in [-blind[1], -blind[0], 0, blind[0], blind[1]]:
    ax.axvline(v, color="k", linestyle=":", linewidth=0.8)
ax.set_ylim(-60, 20)
ax.set_xlabel("Range Rate [m/s]")
ax.set_ylabel("|H|$^2$ [dB]")
ax.set_title(f"MTI canceller response, PRF = {prf / 1e3:.0f} kHz (blind speeds dotted)")
ax.legend()
fig.tight_layout()

# -- Improvement factor vs. clutter spread --
range_rate_stds = np.logspace(-2, 1, 200)
print("\nimprovement factor [dB]:")
fig, ax = plt.subplots()
for order in orders:
    factors = [mti.improvement_factor(order, prf, fcar, s) for s in range_rate_stds]
    ax.semilogx(range_rate_stds, 10 * np.log10(factors), label=f"order {order}")
    at_1 = 10 * np.log10(mti.improvement_factor(order, prf, fcar, 1.0))
    at_10 = 10 * np.log10(mti.improvement_factor(order, prf, fcar, 10.0))
    print(f"  order {order}: {at_1:.1f} dB at 1 m/s, {at_10:.1f} dB at 10 m/s")
ax.set_xlabel("Clutter range-rate std [m/s]")
ax.set_ylabel("Improvement factor [dB]")
ax.set_title("MTI improvement factor, Gaussian clutter spectrum")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

plt.show()
