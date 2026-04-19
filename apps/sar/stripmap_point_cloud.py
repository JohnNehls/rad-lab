#!/usr/bin/env python
"""Stripmap SAR point-cloud example: rendering "rad-lab" as text in a SAR image.

Demonstrates how a *point cloud* — many point scatterers arranged in a
pattern — produces a recognizable shape in a focused SAR image.  Each
letter of "rad-lab" is defined on a 3×5 pixel grid, then mapped to
physical target coordinates.

Parameter design summary:
  - λ = 0.03 m (10 GHz carrier)
  - Azimuth Nyquist: pulse_spacing = v/prf = 0.0125 m < λ/2 = 0.015 m
  - Range resolution: c/(2*bw) = 30 m
  - Cross-range resolution: λ*R/(2*L) ≈ 1.06 m (at ~7 km slant range)
  - Datacube: 1250 range bins × 8000 pulses
  - aperture_length = 100 m (wider than default to fit 7 characters)
  - Cross-range pixel pitch: 3 m (≫ 1.06 m resolution)
  - Range pixel pitch: 60 m (2× the 30 m resolution)
  - ~70 point targets total, all at z = 0, rcs = 10 m²
"""

import matplotlib.pyplot as plt
from rad_lab import sar, SarRadar, SarTarget, lfm_waveform

# -- Pixel font (3 wide × 5 tall) ------------------------------------------------
# Each letter is a list of (row, col) tuples for filled pixels.
# Row 0 is top, col 0 is left.
FONT = {
    "r": [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 2)],
    "a": [(0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 2), (4, 0), (4, 2)],
    "d": [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2), (4, 0), (4, 1)],
    "-": [(2, 0), (2, 1), (2, 2)],
    "l": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2)],
    "b": [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (3, 0), (3, 2), (4, 0), (4, 1)],
}


def text_targets(
    text: str,
    cross_range_pitch: float,
    range_pitch: float,
    nominal_y: float,
    rcs: float,
) -> list[SarTarget]:
    """Convert a text string into a list of SarTargets using the pixel font.

    Letters are centred at x = 0 (cross-range) and arranged around
    *nominal_y* in the range direction.  Row 0 (top of the letter) maps
    to larger y (farther slant range) so the text reads naturally in the
    SAR image where range increases upward on the y-axis.
    """
    char_width = 3  # pixels per character
    char_gap = 1  # pixel gap between characters
    n_rows = 5

    total_cols = len(text) * char_width + (len(text) - 1) * char_gap
    x_offset = -total_cols * cross_range_pitch / 2  # centre the text

    targets: list[SarTarget] = []
    for char_idx, ch in enumerate(text):
        col_origin = char_idx * (char_width + char_gap)
        for row, col in FONT[ch]:
            x = x_offset + (col_origin + col) * cross_range_pitch
            # row 0 → top of letter → larger y (farther range)
            y = nominal_y + (n_rows - 1 - row) * range_pitch
            targets.append(SarTarget(position=[x, y, 0.0], rcs=rcs))
    return targets


# -- Waveform --
bw = 5e6  # bandwidth [Hz] → range resolution = c/(2*bw) = 30 m
waveform = lfm_waveform(bw, T=10e-6, chirp_up_down=1)

# -- SAR system parameters --
sar_radar = SarRadar(
    fcar=10e9,
    tx_power=1e3,
    tx_gain=10 ** (30 / 10),
    rx_gain=10 ** (30 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (8 / 10),
    total_losses=10 ** (8 / 10),
    prf=8000,
    platform_velocity=100,
    aperture_length=100,  # wider aperture to fit 7 characters
    platform_altitude=5e3,
)

# -- Generate point-cloud targets spelling "rad-lab" --
targets = text_targets(
    text="rad-lab",
    cross_range_pitch=3.0,  # 3 m between pixels in cross-range (res ≈ 1.06 m)
    range_pitch=60.0,  # 60 m between pixels in range (res = 30 m)
    nominal_y=5e3,  # centre of text at y = 5 km
    rcs=10.0,
)

# -- Generate the focused SAR image --
cross_range, slant_range, total_image = sar.gen(
    sar_radar,
    waveform,
    targets,
    seed=0,
    plot=False,
)

# -- Plot zoomed to the text region --
fig, ax = sar.plot_sar_image(cross_range, slant_range, total_image, "Focused SAR Image")
# Zoom to the region containing the text (±500 m margin around the 300 m letter height)
text_slant_centre = (5e3**2 + 5e3**2) ** 0.5  # ~7071 m
ax.set_ylim((text_slant_centre - 500) / 1e3, (text_slant_centre + 500) / 1e3)
plt.show()
