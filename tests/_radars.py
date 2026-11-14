"""Shared X-band radar for the RDM tests.

Most tests use the same 10 GHz radar: 1 kW, 30 dB antennas, 8 dB noise
figure and losses, sampled at 20 MHz (Nyquist for a 10 MHz waveform).  They
differ only in PRF, dwell, and the occasional override (``op_temp=0`` for a
noiseless datacube, ``pri_stagger``).
"""

from rad_lab import Radar


def x_band_radar(prf: float, dwell_time: float, **overrides) -> Radar:
    """The shared test radar at ``prf`` and ``dwell_time``; ``overrides`` replace any field."""
    params = dict(
        fcar=10e9,
        tx_power=1e3,
        tx_gain=10 ** (30 / 10),
        rx_gain=10 ** (30 / 10),
        op_temp=290,
        sample_rate=20e6,
        noise_factor=10 ** (8 / 10),
        total_losses=10 ** (8 / 10),
        prf=prf,
        dwell_time=dwell_time,
    )
    return Radar(**{**params, **overrides})
