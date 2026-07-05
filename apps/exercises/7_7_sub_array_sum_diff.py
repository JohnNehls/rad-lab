#!/usr/bin/env python
"""Sub-array sum and difference beam patterns for monopulse.

Split a 20-element ULA into left and right halves (sub-arrays), compute each
sub-array's gain pattern, then form the sum (Sigma) and difference (Delta)
beams.

The sum beam has maximum gain at boresight — used for detection and tracking.
The difference beam has a null at boresight — used for angle estimation.
The monopulse ratio (Delta/Sigma) gives a steep, monotonic curve near
boresight that maps directly to target angle.
"""

import numpy as np
import matplotlib.pyplot as plt
import rad_lab.uniform_linear_arrays as ula
import rad_lab.constants as c


plt.rcParams["text.usetex"] = True

# -- Array parameters --
fc = 10e9  # carrier frequency [Hz]
wavelength = c.C / fc
Nel = 20  # total number of elements
dx = wavelength / 2  # element spacing [m] (half-wavelength)

# -- Create element positions centered about the origin --
L = (Nel - 1) * dx  # total array length [m]
el_pos = np.linspace(-L / 2, L / 2, Nel)

# -- Split into left and right sub-arrays --
assert Nel % 2 == 0
sub_arrays = [el_pos[: int(Nel / 2)], el_pos[int(Nel / 2) :]]

# -- Compute each sub-array's complex gain pattern --
sub_array_gain = []
for sub_array in sub_arrays:
    thetas, gain = ula.linear_antenna_gain_meters(sub_array, fc)
    sub_array_gain.append(gain)

# -- Form sum and difference beams --
sum_gain = sub_array_gain[0] + sub_array_gain[1]  # Sigma: max at boresight
diff_gain = sub_array_gain[0] - sub_array_gain[1]  # Delta: null at boresight

# The sum beam should peak at boresight; the difference beam should have a
# null there with its two peaks symmetric about it.
i_boresight = np.argmin(abs(thetas))
print(f"sum beam:  peak = {abs(sum_gain).max():.2f} at {thetas[np.argmax(abs(sum_gain))]:.2f} deg")
print(
    f"diff beam: peak = {abs(diff_gain).max():.2f} at {abs(thetas[np.argmax(abs(diff_gain))]):.2f} deg (either side of boresight)"
)
print(f"diff beam gain at boresight = {abs(diff_gain[i_boresight]):.3f} (null)")

# -- Plot --
plt.title("Sum and Difference Gain Profiles")
plt.plot(thetas, abs(sum_gain), label=r"$\Sigma$")
plt.plot(thetas, abs(diff_gain), label=r"$\Delta$")
plt.xlabel(r"Angle $\theta$ [deg]")
plt.ylabel("Gain")
plt.xlim((-20, 20))
plt.grid()
plt.legend()
plt.tight_layout()

plt.show()
