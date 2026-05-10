# SAR Image Formation Algorithm

This document describes the synthetic aperture radar (SAR) image formation
algorithm implemented in `sar.gen`. The processor follows the **Range-Doppler
Algorithm** (RDA), one of the most widely used SAR focusing techniques
[1, Ch. 6], [2, Ch. 10].

## Coordinate System

The scene uses a right-handed Cartesian frame:

- $x$ — along-track (flight direction)
- $y$ — cross-track (range direction)
- $z$ — altitude

The platform flies a straight, level path along $x$ at altitude $z_p$ and
constant velocity $v_p$. Targets are modelled as point scatterers with
position $\mathbf{p}_t = [x_t,\, y_t,\, z_t]$ and radar cross section
$\sigma$.

## Algorithm Steps

### 1. Flight Path Generation

The platform occupies $N_p$ equally spaced positions along the synthetic
aperture of length $L_{\text{sa}}$, centred at the origin:

$$
x_n = \left(n - \frac{N_p}{2}\right) \Delta x, \qquad n = 0, \ldots, N_p - 1
$$

where the pulse spacing is $\Delta x = v_p / f_{\text{prf}}$ and the number of
pulses is $N_p = \lceil L_{\text{sa}} / \Delta x \rceil$. Each platform
position is $\mathbf{p}_n = [x_n,\, 0,\, z_p]$.

### 2. Raw Data Generation

For each pulse $n$ and target $t$, the instantaneous slant range is

$$
R_n^{(t)} = \left\| \mathbf{p}_n - \mathbf{p}_t \right\|
$$

The two-way propagation delay and carrier phase are

$$
\tau_n^{(t)} = \frac{2\, R_n^{(t)}}{c}, \qquad
\phi_n^{(t)} = -2\pi f_c\, \tau_n^{(t)}
$$

where $f_c$ is the carrier frequency and $c$ is the speed of light.

The received signal for pulse $n$ is formed by injecting a scaled, phase-shifted
copy of the transmitted waveform $s(t)$ into the datacube at the range bin
corresponding to $\tau_n^{(t)}$:

$$
d\!\left[\,k_n^{(t)},\; n\,\right]
\mathrel{+}= \sqrt{\sigma_t}\; s\!\left[k - k_n^{(t)}\right]\;
e^{\,j\,\phi_n^{(t)}}
$$

where $k_n^{(t)} = \text{round}\!\left(\tau_n^{(t)} f_s\right) - 1$ is the
starting fast-time sample index and $f_s$ is the ADC sampling rate.

**Beam weighting (optional).** In spotlight mode, an amplitude weight is applied
per pulse to model the two-way antenna beam pattern. The pattern is radially
symmetric — it depends only on the scalar off-boresight angle $\theta$, with
no distinction between azimuth and elevation. The default is a Gaussian with
one-way 3 dB beamwidth $\theta_{\text{bw}}$:

$$
w_n^{(t)} = \exp\!\left(-4 \ln 2 \left(\frac{\theta_n^{(t)}}{\theta_{\text{bw}}}\right)^{\!2}\right)
$$

where $\theta_n^{(t)}$ is the off-boresight angle from the steered beam centre
to the target at pulse $n$. The amplitude factor $\sqrt{\sigma_t}$ is then
replaced by $\sqrt{\sigma_t}\, w_n^{(t)}$.

### 3. Noise Addition

Receiver thermal noise is added to the datacube. Each sample is drawn from a
uniform distribution scaled to the expected noise voltage:

$$
v_{\text{noise}} = \sqrt{R_L \cdot k_B\, T_{\text{op}}\, B_w\, F_n}
$$

where $R_L$ is the radar load impedance, $k_B$ is Boltzmann's constant,
$T_{\text{op}}$ is the operating temperature, $B_w$ is the waveform bandwidth,
and $F_n$ is the receiver noise factor.

### 4. Range Compression

Each pulse (column of the datacube) is compressed by convolving with the
matched filter — the time-reversed, conjugated transmit waveform $s^*(-t)$.
This is implemented efficiently via FFT convolution [1, Ch. 4]:

$$
d_{\text{rc}}[k, n] = d[k, n] \ast s^*[-k]
= \mathcal{F}^{-1}\!\Big\{\mathcal{F}\{d[\cdot, n]\}\;\mathcal{F}\{s^*[-\cdot]\}\Big\}
$$

Range compression concentrates each target's energy into a peak whose width
is set by the waveform's range resolution $\delta_r = c / (2B_w)$.

### 5. Range Cell Migration Correction

After range compression each target's energy lives at the range bin
corresponding to its instantaneous slant range $R_n^{(t)}$. Because $R$
varies hyperbolically with slow-time across the aperture, the target's
energy traces a curve in the $(R,\, x_n)$ plane rather than sitting in a
single range row. When the total migration

$$
\Delta R_{\max} \approx \frac{L_{\text{sa}}^2}{8\, R_0}
$$

exceeds a fraction of a range cell, the azimuth matched filter at any
single bin collects only part of the target's energy and the focused peak
defocuses [1, Ch. 6]. **Range Cell Migration Correction (RCMC)** realigns
the trajectory to a constant range $R_0$ before azimuth focusing.

**Migration in the range-Doppler domain.** RCMC operates after an azimuth
FFT, in the range-Doppler domain $(R,\, f_\eta)$. By the principle of
stationary phase, a target at closest-approach range $R_0$ contributes to
azimuth-Doppler bin $f_\eta$ only at the slow-time $\eta^\ast$ where its
instantaneous Doppler matches $f_\eta$ [1, Ch. 6]:

$$
f_\eta
= -\frac{2}{\lambda}\,\frac{\mathrm{d}R}{\mathrm{d}\eta}
= -\frac{2\, v_p^2\, \eta^\ast}{\lambda\, R(\eta^\ast)}
$$

Solving for $R$ as a function of $f_\eta$ gives the closed-form migration

$$
R(f_\eta;\, R_0)
= \frac{R_0}{\sqrt{1 - \left(\dfrac{\lambda\, f_\eta}{2\, v_p}\right)^{\!2}}},
\qquad
\Delta R(f_\eta;\, R_0)
= R_0\!\left[\frac{1}{\sqrt{1 - (\lambda f_\eta / 2 v_p)^2}} - 1\right].
$$

The familiar parabolic approximation
$\Delta R \approx \lambda^{2}\, R_0\, f_\eta^{2}\,/\,(8\, v_p^{2})$
follows from Taylor-expanding to second order in $\lambda f_\eta/(2 v_p)$.

**Correction.** For each range bin $k$ with $R_0 = R_{\text{axis}}[k]$,
the value at azimuth-Doppler bin $m$ is replaced by the sample at the
migrated row:

$$
d_{\text{rcmc}}[k,\, m]
= d_{\text{rd}}\!\left[\,k + \frac{\Delta R(f_{\eta,m};\, R_0)}{\delta_r},\; m\,\right]
$$

where $\delta_r = c / (2 f_s)$ is the range-bin spacing. The fractional row
index calls for interpolation; the implementation uses an 8-tap
Hann-windowed sinc kernel, a standard choice for the RDA accurate to
roughly $-40$ dB for sub-cell shifts [1, §6.3]. An inverse azimuth FFT
then returns the data to slow-time for the windowing and azimuth-focusing
steps that follow.

**Aliasing and small-squint regime.** The formula assumes the target's
Doppler bandwidth $B_d = 2 v_p L_{\text{sa}} / (R_0 \lambda)$ fits below
the PRF; otherwise the parabolic trajectory wraps around in $f_\eta$ and
RCMC corrects the wrong row. Likewise the stationary-phase derivation
assumes $L_{\text{sa}} / R_0 \ll 1$, so very short ranges or very long
apertures need an exact range-migration algorithm (e.g.\ omega-K) instead.

### 6. Azimuth Windowing

A window function $w[n]$ (default: Chebyshev, 60 dB sidelobes) is applied
along the slow-time axis before azimuth compression to reduce cross-range
sidelobes at the cost of slightly wider mainlobe width:

$$
d_{\text{win}}[k, n] = d_{\text{rcmc}}[k, n] \cdot w[n]
$$

### 7. Azimuth Compression (Focusing)

The azimuth matched filter exploits the hyperbolic range history of a point
target. For a target at broadside range $R_0$ (the closest-approach slant
range for range bin $k$), the range history as a function of along-track
position is [1, Ch. 6], [2, Ch. 9]:

$$
R(x_n) = \sqrt{R_0^2 + x_n^2}
$$

The reference signal for range bin $k$ is the phase that a point target at
broadside range $R_0 = R_{\text{axis}}[k]$ would produce:

$$
h_{\text{ref}}[n] = \exp\!\left(-j\,\frac{4\pi}{\lambda}\,R(x_n)\right)
$$

where $\lambda = c / f_c$ is the carrier wavelength. Azimuth focusing is
performed by correlating each range bin with its reference signal via FFT:

$$
d_{\text{focused}}[k, n]
= \mathcal{F}^{-1}\!\Big\{
\mathcal{F}\{d_{\text{win}}[k, \cdot]\}\;
\mathcal{F}\{h_{\text{ref}}\}^*
\Big\}
$$

After an `fftshift` to centre the zero-Doppler bin, the result is the focused
SAR image in range $\times$ cross-range coordinates.

### 8. Cross-Range Axis

The cross-range axis maps directly to the along-track aperture positions:

$$
y_{\text{cr}}[n] = x_n = \left(n - \frac{N_p}{2}\right) \Delta x
$$

## Resolution

- **Range resolution:** $\delta_r = \dfrac{c}{2 B_w}$, determined by the
  waveform bandwidth.
- **Stripmap cross-range resolution:** $\delta_{\text{cr}} = \dfrac{\lambda\, R_0}{2\, L_{\text{sa}}}$,
  improving with longer apertures [1, Ch. 3].
- **Spotlight cross-range resolution:** $\delta_{\text{cr}} = \dfrac{\lambda}{2\,\Delta\theta}$,
  where $\Delta\theta$ is the total angular extent of the synthetic aperture
  [2, Ch. 3].

## References

1. I. G. Cumming and F. H. Wong, *Digital Processing of Synthetic Aperture
   Radar Data: Algorithms and Implementation*. Norwood, MA: Artech House, 2005.
2. M. Soumekh, *Synthetic Aperture Radar Signal Processing with MATLAB
   Algorithms*. New York: Wiley, 1999.
