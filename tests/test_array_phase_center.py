import numpy as np
import pytest
import rad_lab.uniform_linear_arrays as ula


def test_uniform_weights_center():
    # Uniform weights → phase center at mean of positions
    positions = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert ula.array_phase_center(positions, weights) == pytest.approx(1.0)


def test_single_element():
    # Single element → phase center at that element's position
    positions = np.array([5.0])
    weights = np.array([1.0])
    assert ula.array_phase_center(positions, weights) == pytest.approx(5.0)


def test_asymmetric_weights_shift():
    # Heavier weight on left → phase center shifts left of center
    positions = np.array([0.0, 1.0, 2.0])
    weights = np.array([2.0, 1.0, 0.0])
    # sum(|w|*pos) / sum(|w|) = (0 + 1 + 0) / 3 = 1/3
    assert ula.array_phase_center(positions, weights) == pytest.approx(1.0 / 3.0)


def test_symmetric_weights_symmetric_positions():
    # Symmetric array with symmetric weights → phase center at geometric center
    positions = np.array([-1.0, 0.0, 1.0])
    weights = np.array([2.0, 1.0, 2.0])
    # sum(|w|*pos) = 2*(-1) + 1*0 + 2*1 = 0 → phase center at 0
    assert ula.array_phase_center(positions, weights) == pytest.approx(0.0)


def test_weight_phase_does_not_shift_center():
    # Element phase (sign or complex rotation) steers the beam but must not
    # move the aperture centroid — only weight magnitudes matter.
    positions = np.array([0.0, 1.0, 2.0])
    flipped = np.array([1.0, -1.0, 1.0])
    steered = np.exp(1j * np.array([0.0, 0.5, 1.0]))
    assert ula.array_phase_center(positions, flipped) == pytest.approx(1.0)
    assert ula.array_phase_center(positions, steered) == pytest.approx(1.0)
