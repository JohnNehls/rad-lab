# Range-Doppler Map (RDM) Generation Algorithm

This document describes the range-Doppler map generation algorithm implemented
in `rdm.gen`. The processor follows the standard **pulse-Doppler processing**
chain: populate a datacube with target returns, apply a matched filter for
pulse compression, window the slow-time dimension, and Doppler-process with
an FFT [1, Ch. 5], [2, Ch. 3].

## Signal Model

The radar transmits $N_p$ pulses at a pulse repetition frequency
$f_{\text{prf}}$ over a coherent processing interval (CPI) of duration
$T_{\text{cpi}}$, where $N_p = \lceil T_{\text{cpi}} \cdot f_{\text{prf}} \rceil$.

Each target is a point scatterer characterised by:

- Range $R$ [m] — distance from the radar at the start of the CPI
- Range rate $\dot{R}$ [m/s] — radial velocity (positive = receding)
- Radar cross section $\sigma$ [m$^2$]
- Steering vector component $\mathbf{s}$ (for array processing; defaults to 1)

The datacube $d[k, n]$ has dimensions $N_r \times N_p$ (fast-time range bins
$\times$ slow-time pulses), where $N_r = f_s / f_{\text{prf}}$ and $f_s$ is
the ADC sampling rate.

## Algorithm Steps

### 1. Datacube Population — Skin Returns

For each pulse $n = 0, \ldots, N_p - 1$ the pulse transmit time is

$$
t_n = \frac{n}{f_{\text{prf}}}
$$

The instantaneous range to target $t$ at pulse $n$ is

$$
R_n^{(t)} = R^{(t)} + \dot{R}^{(t)}\, t_n
$$

The two-way propagation delay and carrier phase are

$$
\tau_n^{(t)} = \frac{2\, R_n^{(t)}}{c}, \qquad
\phi_n^{(t)} = -2\pi f_c\, \tau_n^{(t)}
$$

where $f_c$ is the carrier frequency and $c$ is the speed of light.

The received signal voltage for a skin return is computed using the two-way
radar range equation [1, Ch. 2]:

$$
P_r = \frac{P_t\, G_t\, G_r\, \sigma\, \lambda^2}{(4\pi)^3\, R^4\, L}
$$

$$
V_r = \sqrt{R_L \cdot P_r}
$$

where $P_t$ is transmit power, $G_t$ and $G_r$ are transmit and receive gains,
$\lambda = c / f_c$ is the carrier wavelength, $L$ represents system losses,
and $R_L$ is the radar load impedance.

A scaled, phase-shifted copy of the transmitted waveform $s[k]$ is injected
into the datacube at the appropriate range bin:

$$
d\!\left[\,k_n^{(t)},\; n\,\right]
\mathrel{+}= V_r\, \mathbf{s}^{(t)}\; s\!\left[k - k_n^{(t)}\right]\;
e^{\,j\,\phi_n^{(t)}}
$$

where the starting sample index is

$$
k_n^{(t)} = \text{round}\!\left(\tau_n^{(t)}\, f_s\right) - 1
$$

### 2. Datacube Population — Jammer (DRFM) Returns

When an electronic attack (EA) platform is present, a digital radio frequency
memory (DRFM) jammer intercepts the radar pulse, stores it, and retransmits
it with modulation designed to deceive the radar [3].

The jammer return amplitude uses the one-way Friis transmission equation since
the jammer actively retransmits rather than passively reflecting:

$$
P_{r,\text{jam}} = \frac{P_{t,\text{jam}}\, G_{t,\text{jam}}\, G_r\, \lambda^2}
{(4\pi)^2\, R^2\, L_{\text{jam}}}
$$

The DRFM applies several modulation effects to the stored pulse:

- **Range offset** $\Delta R$: an additional two-way time delay
  $2\Delta R / c$ shifts the return to a false range bin.
- **Doppler offset** $\Delta\dot{R}$: a per-pulse phase ramp
  $e^{-j\, 2\pi n\, f_{\Delta d} / f_{\text{prf}}}$ where
  $f_{\Delta d} = 2 f_c \Delta\dot{R} / c$ shifts the return in the
  Doppler dimension.
- **Velocity bin masking (VBM)**: a slow-time phase modulation
  $\psi_{\text{vbm}}[n]$ spreads jammer energy across multiple Doppler
  bins to mask the true target velocity [4].

The composite jammer return at pulse $n$ is

$$
d_{\text{jam}}[k_n, n] \mathrel{+}=
V_{r,\text{jam}}\, \mathbf{s}^{(t)}\,
\psi_{\text{vbm}}[n]\,
e^{\,j\, n\, \hat{\phi}_d}\,
e^{-j\, 2\pi n\, f_{\Delta d} / f_{\text{prf}}}\,
s_{\text{stored}}[k - k_n]\,
e^{\,j\,\phi_{1\text{-way},n}}
$$

where $\hat{\phi}_d$ is the estimated pulse-to-pulse Doppler phase of the
target, $s_{\text{stored}}$ is the intercepted pulse captured on the first
PRI, and $\phi_{1\text{-way},n}$ is the one-way return-path propagation phase.

### 3. Noise Addition

Receiver thermal noise is added to the datacube. In voltage mode, each sample
is drawn from a uniform distribution scaled to the noise voltage:

$$
v_{\text{noise}} = \sqrt{R_L \cdot k_B\, T_{\text{op}}\, B_w\, F_n}
$$

where $k_B$ is Boltzmann's constant, $T_{\text{op}}$ is the operating
temperature, $B_w$ is the waveform bandwidth, and $F_n$ is the receiver noise
factor.

In SNR mode, the noise floor is normalised to unity variance using complex
Gaussian noise scaled by $1/\sqrt{N_p}$, so that target amplitudes read
directly as post-integration SNR voltage ratios.

### 4. Range Compression (Matched Filtering)

Each pulse (column of the datacube) is compressed by convolving with the
matched filter — the time-reversed, conjugated transmit waveform $s^*(-t)$.
This is implemented via FFT convolution [1, Ch. 4]:

$$
d_{\text{rc}}[k, n] = d[k, n] \ast s^*[-k]
= \mathcal{F}^{-1}\!\Big\{\mathcal{F}\{d[\cdot, n]\}\;\mathcal{F}\{s^*[-\cdot]\}\Big\}
$$

The matched filter maximises the output SNR and compresses each target's energy
into a peak whose width is determined by the waveform's range resolution
$\delta_r = c / (2B_w)$. For coded waveforms the processing gain equals the
time-bandwidth product $\tau B_w$ [1, Ch. 4].

### 5. Doppler Windowing

A window function $w[n]$ is applied along the slow-time axis before the
Doppler FFT to suppress spectral sidelobes at the cost of slightly wider
mainlobe width [1, Ch. 4]:

$$
d_{\text{win}}[k, n] = d_{\text{rc}}[k, n] \cdot w[n]
$$

The window is normalised so that $\text{mean}(w) = 1$, preserving coherent
gain. Available windows are:

| Window | Typical sidelobe level | Notes |
|--------|----------------------|-------|
| Chebyshev (default) | $-60$ dB | Equi-ripple; configurable via `at` parameter |
| Blackman-Harris | $\approx -92$ dB | Very low sidelobes, wider mainlobe |
| Taylor | Configurable | Good near-in sidelobe control (`nbar`, `sll`) |
| None (rectangular) | $\approx -13$ dB | Narrowest mainlobe, highest sidelobes |

### 6. Doppler Processing (Slow-Time FFT)

The slow-time FFT transforms each range bin from the time domain into the
Doppler frequency domain [1, Ch. 5], [2, Ch. 3]:

$$
D[k, m] = \sum_{n=0}^{N_p - 1} d_{\text{win}}[k, n]\; e^{-j\,2\pi\, mn / N_p}
$$

followed by an `fftshift` to centre zero Doppler. The resulting Doppler
frequency axis is

$$
f_m = \text{fftshift}\!\left(\text{fftfreq}(N_p,\; 1/f_{\text{prf}})\right)
$$

spanning $[-f_{\text{prf}}/2,\; f_{\text{prf}}/2)$.

### 7. Range-Rate Axis

The Doppler frequency axis is converted to range rate using the standard
Doppler relationship [1, Ch. 2]:

$$
\dot{R}_m = -\frac{c\, f_m}{2\, f_c}
$$

The sign convention is that positive range rate corresponds to a receding
target (negative Doppler shift).

## Resolution and Ambiguities

- **Range resolution:** $\delta_r = \dfrac{c}{2\, B_w}$, determined by the
  waveform bandwidth.
- **Doppler (velocity) resolution:** $\delta_{\dot{R}} = \dfrac{\lambda}{2\, T_{\text{cpi}}}$,
  determined by the CPI duration [1, Ch. 5].
- **Maximum unambiguous range:** $R_{\text{ua}} = \dfrac{c}{2\, f_{\text{prf}}}$.
- **Maximum unambiguous range rate:** $\dot{R}_{\text{ua}} = \pm\dfrac{c\, f_{\text{prf}}}{4\, f_c}$.

## References

1. M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.
   New York: McGraw-Hill, 2014.
2. M. A. Richards, J. A. Scheer, and W. A. Holm, Eds., *Principles of Modern
   Radar: Basic Principles*. Raleigh, NC: SciTech Publishing, 2010.
3. D. C. Schleher, *Electronic Warfare in the Information Age*. Norwood, MA:
   Artech House, 1999.
4. L. Neng-Jing and Z. Yi-Ting, "A survey of radar ECM and ECCM," *IEEE
   Trans. Aerosp. Electron. Syst.*, vol. 31, no. 3, pp. 1110–1120, Jul. 1995.
