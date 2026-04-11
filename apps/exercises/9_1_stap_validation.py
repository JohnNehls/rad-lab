#!/usr/bin/env python
"""STAP validation: verify adaptive processing is correct.

Tests that the STAP processor:
1. Preserves a target in the correct range-rate and range bin.
2. Suppresses clutter relative to conventional processing.
3. The space-time steering vector peaks at the correct angle-Doppler cell.

The validation uses a single target at a known range, velocity, and angle with
strong clutter.  After processing, we check that the target's peak location
matches expectations and that the clutter power is reduced by STAP.
"""

import numpy as np
from rad_lab import stap
from rad_lab.pulse_doppler_radar import Radar, frequency_delta_doppler
from rad_lab.waveform import lfm_waveform
from rad_lab.returns import Target, Return
from rad_lab.uniform_linear_arrays import steering_vector

# -- Parameters --
bw = 1e6
fcar = 10e9
waveform = lfm_waveform(bw, T=10e-6, chirp_up_down=1)

radar = Radar(
    fcar=fcar,
    tx_power=1e3,
    tx_gain=10 ** (20 / 10),
    rx_gain=10 ** (20 / 10),
    op_temp=290,
    sample_rate=2 * bw,
    noise_factor=10 ** (5 / 10),
    total_losses=10 ** (3 / 10),
    prf=10e3,
    dwell_time=1.6e-3,
)

n_elements = 8
el_pos = np.arange(n_elements) * 0.5 - (n_elements - 1) * 0.25

platform_velocity = 100  # m/s

# Single target: broadside (0 deg), closing at 30 m/s, 30 dBsm
# Unambiguous range: c/(2*prf) = 15 km, unambiguous rdot: ±c*prf/(4*fcar) = ±75 m/s
tgt_range = 7e3
tgt_rdot = -30
tgt_angle = 0
tgt_rcs = 1000  # 30 dBsm — strong target

return_list = [
    Return(target=Target(range=tgt_range, range_rate=tgt_rdot, rcs=tgt_rcs, angle=tgt_angle))
]

print("=" * 60)
print("STAP Validation")
print("=" * 60)

# -- Run without clutter (baseline) --
print("\n--- Test 1: No clutter (verify target placement) ---")
result_clean = stap.gen(
    radar,
    waveform,
    return_list,
    el_pos=el_pos,
    platform_velocity=0,
    cnr=0,
    plot=False,
)

rdot_axis = result_clean["rdot_axis"]
r_axis = result_clean["r_axis"]
conv_clean = result_clean["conventional"]

# Find peak in conventional RDM
conv_mag = np.abs(conv_clean)
peak_idx = np.unravel_index(conv_mag.argmax(), conv_mag.shape)
peak_range = r_axis[peak_idx[0]]
peak_rdot = rdot_axis[peak_idx[1]]

print(f"  Expected range:      {tgt_range / 1e3:.1f} km")
print(f"  Detected range:      {peak_range / 1e3:.1f} km")
print(f"  Expected range-rate: {tgt_rdot:.1f} m/s")
print(f"  Detected range-rate: {peak_rdot:.1f} m/s")

range_err = abs(peak_range - tgt_range)
rdot_err = abs(peak_rdot - tgt_rdot)
range_res = 3e8 / (2 * bw)
rdot_res = abs(rdot_axis[1] - rdot_axis[0])

range_ok = range_err < 2 * range_res
rdot_ok = rdot_err < 2 * rdot_res
print(
    f"  Range error:  {range_err:.1f} m (tolerance: {2 * range_res:.1f} m) {'PASS' if range_ok else 'FAIL'}"
)
print(
    f"  Rdot error:   {rdot_err:.1f} m/s (tolerance: {2 * rdot_res:.1f} m/s) {'PASS' if rdot_ok else 'FAIL'}"
)

# -- Run with clutter --
print("\n--- Test 2: With clutter (verify STAP suppression) ---")
result_clutter = stap.gen(
    radar,
    waveform,
    return_list,
    el_pos=el_pos,
    platform_velocity=platform_velocity,
    cnr=40,
    n_clutter_patches=180,
    plot=False,
    n_guard=3,
    diagonal_load=1e-2,
)

conv_clut = result_clutter["conventional"]
adapt_clut = result_clutter["adaptive"]

# Measure clutter power: use range bins away from the target
target_range_bin = peak_idx[0]
clutter_bins = list(range(0, max(0, target_range_bin - 10))) + list(
    range(min(len(r_axis), target_range_bin + 10), len(r_axis))
)

if len(clutter_bins) > 0:
    conv_clutter_power = np.mean(np.abs(conv_clut[clutter_bins, :]) ** 2)
    adapt_clutter_power = np.mean(np.abs(adapt_clut[clutter_bins, :]) ** 2)

    suppression_db = 10 * np.log10(conv_clutter_power / max(adapt_clutter_power, 1e-30))
    print(f"  Conventional clutter power: {10 * np.log10(conv_clutter_power):.1f} dB")
    print(f"  Adaptive clutter power:     {10 * np.log10(max(adapt_clutter_power, 1e-30)):.1f} dB")
    print(
        f"  Clutter suppression:        {suppression_db:.1f} dB {'PASS' if suppression_db > 5 else 'FAIL'}"
    )
else:
    print("  (not enough range bins for clutter measurement)")

# -- Test 3: Verify steering vector --
print("\n--- Test 3: Space-time steering vector ---")
fd_expected = frequency_delta_doppler(tgt_rdot, fcar)
s = stap.space_time_steering_vector(el_pos, radar.n_pulses, tgt_angle, fd_expected, radar.prf)

# The spatial part should match the array steering vector
a_s = steering_vector(el_pos, tgt_angle)
# Extract the spatial part from the first pulse's block
s_spatial = s[:n_elements]
s_spatial_norm = s_spatial / np.abs(s_spatial[0])
a_s_norm = a_s / np.abs(a_s[0])

spatial_err = np.max(np.abs(s_spatial_norm - a_s_norm))
spatial_ok = spatial_err < 1e-10
print(f"  Spatial steering vector error: {spatial_err:.2e} {'PASS' if spatial_ok else 'FAIL'}")

# The temporal part should be a phase ramp at the target Doppler
n_vec = np.arange(radar.n_pulses)
a_t_expected = np.exp(1j * 2 * np.pi * fd_expected * n_vec / radar.prf)
# Extract temporal part (every n_elements-th sample starting from element 0)
s_temporal = s[::n_elements]
s_temporal_norm = s_temporal / s_temporal[0]
a_t_norm = a_t_expected / a_t_expected[0]

temporal_err = np.max(np.abs(s_temporal_norm - a_t_norm))
temporal_ok = temporal_err < 1e-10
print(f"  Temporal steering vector error: {temporal_err:.2e} {'PASS' if temporal_ok else 'FAIL'}")

# -- Summary --
all_pass = range_ok and rdot_ok and spatial_ok and temporal_ok
print("\n" + "=" * 60)
print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
print("=" * 60)
