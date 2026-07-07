#!/usr/bin/env python
"""Array factor via IFFT — parked experiment, incomplete.

Intent: show that a uniform linear array's factor is the (inverse) Fourier
transform of its element weights, so an IFFT of the weight vector traces the
same pattern as the direct sum in ``rad_lab.uniform_linear_arrays``. The
phase-center correction and the angle-axis mapping below are unfinished and
nothing here is validated.

The ``_no_test`` suffix keeps this script out of the apps regression until
it is finished.
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft

Nel = 40  # number of elements
dx = 1 / 2  # element spacing [wavelengths]

weights = np.ones(Nel)  # uniform weighting; try e.g. signal.windows.chebwin(Nel, 60)

# Phase term to reference the pattern to the array center (unverified)
phase = np.exp(1j * 2 * np.pi * (Nel - 1) / 2 * dx)
af = fft.ifftshift(fft.ifft(weights))
af_centered = phase * af

# Element spacing sets the sine-space extent of the pattern
vtheta_max = 1 / (2 * dx)
faxis = np.linspace(-vtheta_max, vtheta_max, Nel)

plt.plot(faxis, abs(af), "-o")
plt.show()

raise NotImplementedError("This IFFT array-factor exercise is incomplete")
