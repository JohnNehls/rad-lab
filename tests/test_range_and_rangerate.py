import pytest
from rad_lab.geometry import range_and_rangerate


def test_problem6_geometry():
    # Example from 04_6_calc_r_and_rdot_from_positions.py
    # Platform at [0, 0, 3048] moving at [300, 0, 0]
    # Target at [5e3, 0, 3048] moving at [-300, 0, 0]
    # Same altitude → R_vec = [5000, 0, 0], R_mag = 5000, R_dot = -600 (closing)
    R_vec, R_mag, R_dot = range_and_rangerate(
        [0, 0, 3048], [300, 0, 0], [5e3, 0, 3048], [-300, 0, 0]
    )
    assert R_mag == pytest.approx(5000.0)
    assert R_dot == pytest.approx(-600.0)


def test_stationary_both():
    # No motion, target directly ahead at 100 m
    _, R_mag, R_dot = range_and_rangerate([0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0])
    assert R_mag == pytest.approx(100.0)
    assert R_dot == pytest.approx(0.0)


def test_closing_target():
    # Stationary platform, target approaching at 50 m/s
    _, R_mag, R_dot = range_and_rangerate([0, 0, 0], [0, 0, 0], [1000, 0, 0], [-50, 0, 0])
    assert R_mag == pytest.approx(1000.0)
    assert R_dot == pytest.approx(-50.0)


def test_opening_target():
    # Stationary platform, target moving away at 50 m/s
    _, R_mag, R_dot = range_and_rangerate([0, 0, 0], [0, 0, 0], [1000, 0, 0], [50, 0, 0])
    assert R_mag == pytest.approx(1000.0)
    assert R_dot == pytest.approx(50.0)


def test_3d_range():
    # 3-4-5 triangle: range should be 500 m
    _, R_mag, R_dot = range_and_rangerate([0, 0, 0], [0, 0, 0], [300, 400, 0], [0, 0, 0])
    assert R_mag == pytest.approx(500.0)
    assert R_dot == pytest.approx(0.0)
