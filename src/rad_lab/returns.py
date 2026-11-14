"""Target, electronic-attack, and clutter return models.

Defines `Target` (kinematics and RCS), `EaPlatform` (DRFM
jammer transmitter parameters), `Return` (pairing of a target with
an optional EA platform), and `Clutter` (distributed ground/sea clutter)
used as inputs to the RDM generator.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass
class Target:
    """Target kinematics and scattering parameters.

    Attributes:
        range: True target range [m].
        range_rate: Radial velocity (positive = receding) [m/s].
        rcs: Radar cross section [m^2].  Required for skin return amplitude;
            unused by jammer return amplitude in physical mode.
        sv: Steering vector component for a single array element [dimensionless].
            Defaults to 1 (isotropic / no array).
    """

    range: float
    range_rate: float
    rcs: float | None = None
    sv: complex = 1


@dataclass
class EaPlatform:
    """Electronic attack (EA) platform — hardware and signal modulation parameters.

    All fields beyond the first three are modulation parameters that control how
    the stored pulse is modified before retransmission.

    Attributes:
        tx_power: Transmit power [W].
        tx_gain: Transmit antenna gain [linear].
        total_losses: Total system losses [linear].
        rdot_delta: VBM Doppler spread [m/s].  ``None`` disables VBM entirely.
        rdot_offset: Doppler frequency offset applied to the retransmitted
            pulse [m/s].  Defaults to 0.
        range_offset: Additional apparent range offset injected by the EA [m].
            Defaults to 0.
        delay: Additional time delay applied to the retransmitted pulse [s].
            Defaults to 0.
        vbm_noise_function: Phase-noise function used for VBM slow-time
            modulation.  ``None`` selects the default LFM noise function.
    """

    tx_power: float
    tx_gain: float
    total_losses: float
    rdot_delta: float | None = None
    rdot_offset: float = 0.0
    range_offset: float = 0.0
    delay: float = 0.0
    vbm_noise_function: Callable | None = None


@dataclass
class Return:
    """A simulated radar return — skin, jammer-based EA, or both simultaneously.

    The two contributions are independent and additive:

    - Skin return fires when ``target.rcs is not None``.  The radar's own
      transmitted pulse reflects off the target.  ``target.rcs`` sets the
      physical amplitude via the two-way range equation.
    - Memory return fires when ``platform is not None``.  A DRFM jammer
      receives the pulse, stores it, and retransmits with the modulation
      specified in ``platform``.  Amplitude is derived from the one-way
      link equation.

    Both can be active on the same Return, modelling a jammer co-located
    with the target (set ``target.rcs`` and supply a ``platform``).

    Attributes:
        target: Target kinematics and (for skin) RCS.
        platform: EA platform parameters.  ``None`` disables jammer return.
    """

    target: Target
    platform: EaPlatform | None = None


@dataclass
class Clutter:
    """Distributed surface clutter over a band of ranges.

    Every range sample in ``[range_start, range_stop]`` gets one clutter cell.
    The cell RCS follows the constant-gamma model,
    ``sigma = gamma * sin(grazing_angle) * A``, with illuminated area
    ``A = R * azimuth_beamwidth * dR / cos(grazing_angle)``.  Each cell's
    slow-time samples are a complex Gaussian process whose Doppler spectrum is
    Gaussian, centred on ``range_rate_mean`` with spread ``range_rate_std``
    (internal motion such as wind-blown vegetation or sea waves).

    Clutter beyond the unambiguous range folds back into the RDM, just like a
    skin return.

    Attributes:
        range_start: Near edge of the clutter band [m].
        range_stop: Far edge of the clutter band [m].
        gamma: Constant-gamma reflectivity [linear], e.g. ``10**(-20/10)``
            for farmland.
        grazing_angle: Grazing angle, assumed constant across the band [rad].
        azimuth_beamwidth: Azimuth beamwidth; sets the cell's cross-range
            width [rad].
        range_rate_mean: Mean clutter radial velocity, e.g. from wind or
            own-ship motion [m/s].  Defaults to 0.
        range_rate_std: Standard deviation of the clutter radial velocity
            (spectral spread) [m/s].  Defaults to 0.1.
    """

    range_start: float
    range_stop: float
    gamma: float
    grazing_angle: float
    azimuth_beamwidth: float
    range_rate_mean: float = 0.0
    range_rate_std: float = 0.1
