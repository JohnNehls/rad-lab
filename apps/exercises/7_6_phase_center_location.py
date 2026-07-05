#!/usr/bin/env python
"""Array phase center calculation.

Compute the phase center of a weighted linear array. The phase center is the
weighted centroid of the element positions — it tells you the effective
"electrical center" of the array, which matters for monopulse and
interferometric processing.

For a symmetric, uniformly-weighted array the phase center is at the geometric
center. Asymmetric weighting shifts it toward the heavier-weighted elements.

Note: the reference document has an error in the phase center equation —
it omits the normalization by the sum of weights.
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import rad_lab.uniform_linear_arrays as ula


plt.rcParams["text.usetex"] = True

# -- Define a 40-element array with Chebyshev weights --
Nel = 40
chebWindow = signal.windows.chebwin(Nel, 30)  # 30 dB Chebyshev weights (un-normalized)
pos_ar = np.linspace(2, 8, Nel) * 1e-3  # element positions [m]

# -- Compute the phase center (weighted centroid of positions) --
phase_cent = ula.array_phase_center(pos_ar, chebWindow)
print(
    f"phase center = {phase_cent * 1e3:.3f} mm (geometric center = {pos_ar.mean() * 1e3:.3f} mm)"
)

# -- Plot: element weights and phase center location --
plt.figure()
plt.title("Array Weights and Phase Center")
plt.plot(pos_ar, chebWindow, "o", label="weights")
plt.axvline(x=phase_cent, linestyle="dashed", color="k", label="phase center")
plt.xlabel("Array Element Position [m]")
plt.ylabel("Weight")
plt.legend()
plt.tight_layout()
plt.show()
