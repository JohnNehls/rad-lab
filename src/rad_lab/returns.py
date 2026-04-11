"""Target and electronic-attack return models.

Defines :class:`Target` (kinematics and RCS), :class:`EaPlatform` (DRFM
jammer transmitter parameters), and :class:`Return` (pairing of a target with
an optional EA platform) used as inputs to the RDM generator.
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
            Used by RDM module to for array element specific responses.
            Defaults to 1 (isotropic / no array).
        angle: Angle of arrival [degrees], 0 = broadside.  Used by the STAP
            module for spatial steering.  Defaults to 0.
    """

    range: float
    range_rate: float
    rcs: float | None = None
    sv: complex = 1
    angle: float = 0.0


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
