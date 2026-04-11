"""Radar Signal Processing (rad_lab) — RDM generation and radar signal processing tools.

Typical usage::

    from rad_lab import rdm, Radar, Target, Return
    from rad_lab import lfm_waveform

    radar = Radar(fcar=10e9, tx_power=1e3, ...)
    waveform = lfm_waveform(bw=10e6, T=1e-6, chirp_up_down=1)
    rdm.gen(radar, waveform, [Return(target=Target(range=3e3, range_rate=1e3, rcs=10))])
"""

from .pulse_doppler_radar import Radar
from .returns import Target, EaPlatform, Return
from .waveform import (
    WaveformType,
    WaveformSample,
    uncoded_waveform,
    barker_coded_waveform,
    random_coded_waveform,
    lfm_waveform,
)
from .sar_radar import SarRadar, SarTarget
from . import ambiguity
from . import detection
from . import cfar
from . import multiprf
from . import rdm
from . import sar
from . import stap
from ._version import __version__

__all__ = [
    "Radar",
    "Target",
    "EaPlatform",
    "Return",
    "SarRadar",
    "SarTarget",
    "WaveformType",
    "WaveformSample",
    "uncoded_waveform",
    "barker_coded_waveform",
    "random_coded_waveform",
    "lfm_waveform",
    "ambiguity",
    "detection",
    "cfar",
    "multiprf",
    "rdm",
    "sar",
    "stap",
    "__version__",
]
