import numpy as np
from scipy import fft
from . import constants as c
from .range_equation import snr_range_eqn_cp
from .pulse_doppler_radar import Radar
from .waveform import WaveformSample
from .returns import Target


def noise_checks(noise_dc: np.ndarray, total_dc: np.ndarray) -> None:
    """Prints noise and total-signal statistics for debugging.

    Args:
        noise_dc (np.ndarray): Pre-processing noise-only datacube.
        total_dc (np.ndarray): Post-processing (signal + noise) datacube.
    """
    print(f"\n5.3.2 noise check: {np.var(fft.fft(noise_dc, axis=1))=: .4f}")
    print("\nnoise check:")
    noise_var = np.var(total_dc, 1)
    print(f"\t{np.mean(noise_var)=: .4f}")
    print(f"\t{np.var(noise_var)=: .4f}")
    print(f"\t{np.mean(20*np.log10(noise_var))=: .4f}")
    print(f"\t{np.var(20*np.log10(noise_var))=: .4f}")
    print("\nPeak magnitude:")
    print(f"\t{20*np.log10(np.max(abs(noise_dc)))=:.2f}")
    print(f"\t{20*np.log10(np.max(abs(total_dc)))=:.2f}")


def verify_snr(snr_dc: np.ndarray, radar: Radar, target: Target, waveform: WaveformSample) -> None:
    """Compare the measured peak SNR in an SNR-normalised RDM to theory.

    Prints the RDM peak and the range-equation prediction side by side,
    plus the off-peak noise-floor std (which should be ~1 if the SNR
    normalisation is right).  Doppler windowing perturbs the noise
    variance by a small factor (Parseval on the window coefficients),
    so expect small deviations from unity.

    Args:
        snr_dc: RDM in SNR voltage ratio (output of :func:`rad_lab.rdm.to_snr`).
        radar: Radar system parameters.
        target: Target whose return is expected to dominate the RDM.
        waveform: WaveformSample used for the simulation.
    """
    magnitude = np.abs(snr_dc)
    measured_db = 20 * np.log10(magnitude.max())
    expected_linear = snr_range_eqn_cp(
        radar.tx_power,
        radar.tx_gain,
        radar.rx_gain,
        target.rcs,
        c.C / radar.fcar,
        target.range,
        waveform.bw,
        radar.noise_factor,
        radar.total_losses,
        radar.op_temp,
        radar.n_pulses,
        waveform.time_bw_product,
    )
    expected_db = 10 * np.log10(expected_linear)

    # Noise floor: complex-gaussian variance -> voltage RMS of |X| is sqrt(2*sigma^2)
    # For an SNR-normalised RDM we expect per-cell complex variance = 1, so RMS(|X|) = 1.
    # Exclude a ±5-cell box around the peak so the signal lobe doesn't bias the estimate.
    peak_r, peak_c = np.unravel_index(magnitude.argmax(), magnitude.shape)
    r0, r1 = max(0, peak_r - 5), min(magnitude.shape[0], peak_r + 6)
    c0, c1 = max(0, peak_c - 5), min(magnitude.shape[1], peak_c + 6)
    mask = np.ones_like(magnitude, dtype=bool)
    mask[r0:r1, c0:c1] = False
    noise_rms = np.sqrt(np.mean(magnitude[mask] ** 2))

    print("SNR verification:")
    print(f"  Measured peak:  {measured_db:6.2f} dB   (RDM)")
    print(f"  Range equation: {expected_db:6.2f} dB   (theory)")
    print(f"  Difference:     {measured_db - expected_db:+6.2f} dB")
    print(f"  Noise RMS:      {noise_rms:6.3f}      (voltage ratio, expect ~1)")
