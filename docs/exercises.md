# rad-lab Exercises

Each exercise below lists the problem statement, required parameters, and
expected figures.  Solutions are in `apps/exercises/`.

---

## 1 — Range Equation

### Exercise 1.0: Range Equation Studies

**Problem 1 — BPSK SNR vs Range.**
Compute and plot the post-integration SNR as a function of range for a
Barker-13 BPSK waveform.  Produce a 3-subplot figure (one subplot per
transmit power) with three curves per subplot (one per RCS value) and a
horizontal detection threshold.

| Parameter              | Value                              |
|------------------------|------------------------------------|
| Antenna diameter       | $1.5 \times 12 \times 0.0254$ m (1.5 ft) |
| Carrier frequency      | 10 GHz                             |
| Bandwidth              | 10 MHz                             |
| Code length            | 13 (Barker)                        |
| System losses          | 8 dB                               |
| Noise factor           | 6 dB                               |
| System temperature     | 290 K                              |
| Number of pulses       | 256                                |
| Transmit powers        | 1, 5, 10 kW                        |
| Target RCS             | 0, 10, 20 dBsm                     |
| Range sweep            | 1 km to 30 km, step 100 m          |
| Detection threshold    | 12 dB                              |

**Figure:** 3 subplots (one per $P_t$).  Each subplot: SNR [dB] vs range [km]
with three RCS curves and a 12 dB threshold line.

**Problem 2 — Duty-Factor SNR vs Range.**
Repeat the SNR-vs-range calculation using the duty-factor form of the range
equation.  Plot for three CPI durations and three duty factors.

| Parameter              | Value                              |
|------------------------|------------------------------------|
| Target RCS             | 0 dBsm                            |
| Transmit power         | 5 kW                              |
| CPI durations          | 2, 5, 10 ms                       |
| Duty factors           | 0.01, 0.1, 0.2                    |

**Figure:** 3 subplots (one per CPI duration).  Each subplot: SNR [dB] vs
range [km] with three duty-factor curves.

**Problem 3 — Minimum Detectable Range Heatmaps.**
Compute the minimum detectable range over a 2-D parameter grid and display
as a heatmap.

| Parameter              | Value                              |
|------------------------|------------------------------------|
| Detection threshold    | 15 dB                              |
| Duty factor            | 0.1                               |
| CPI duration           | 2 ms                              |

Heatmap A:

| Axis         | Range                     |
|--------------|---------------------------|
| Transmit power | 500 W to 10 kW, step 100 W |
| Target RCS     | $-5$ to 25 dBsm, step 1 dB |

Heatmap B:

| Axis         | Range                     |
|--------------|---------------------------|
| CPI duration | 1 ms to 50 ms, step 200 $\mu$s |
| Target RCS   | $-5$ to 25 dBsm, step 1 dB |

**Figures:** Two pcolormesh heatmaps of minimum detectable range [km].

---

## 2 — Pulse-Doppler Radar Ambiguities

### Exercise 2.0: Ambiguity Studies

**Problem 1 — Unambiguous Range vs PRF.**
Plot the maximum unambiguous range as a function of PRF.

| Parameter | Value                    |
|-----------|--------------------------|
| PRF sweep | 1 kHz to 200 kHz, step 500 Hz |

**Figure:** Unambiguous range [km] vs PRF [kHz], log-scale y-axis.

**Problem 2 — Unambiguous Range Rate vs PRF.**
Plot the maximum unambiguous range rate for multiple carrier frequencies.

| Parameter          | Value                                    |
|--------------------|------------------------------------------|
| Carrier frequencies | 1, 2, 3, 4, 12, 16, 35, 95 GHz         |

**Figure:** $\pm$ max range rate [km/s] vs PRF [kHz], log-scale y-axis, one
curve per carrier frequency.

**Problem 3 — Range Aliasing.**
Show how a target's apparent range aliases as PRF changes.

| Parameter    | Value                                              |
|--------------|----------------------------------------------------|
| True range   | 15.5 km                                           |
| PRF values   | 2, 4, 8, 16, 32, 50, 60, 64, 95, 100, 128, 150, 228 kHz |

**Figure:** Apparent range [km] vs PRF [kHz] with unambiguous range overlay.

**Problem 4 — Doppler and Range-Rate Aliasing vs PRF.**
Show Doppler and range-rate aliasing for a target at a fixed velocity.

| Parameter        | Value       |
|------------------|-------------|
| Target range rate | $-750$ m/s |
| Carrier frequency | 10 GHz     |

**Figure:** 2 subplots — apparent Doppler [kHz] vs PRF [kHz], and apparent
range rate [m/s] vs PRF [kHz], each with unambiguous bounds.

**Problem 5 — Range-Rate Aliasing vs Carrier Frequency.**
Show how the same target velocity aliases differently at different frequencies.

| Parameter        | Value                                         |
|------------------|-----------------------------------------------|
| Target range rate | $-750$ m/s                                   |
| PRF              | 16 kHz                                        |
| Carrier frequencies | 1, 2, 4, 6, 8, 10, 12, 16, 18, 34, 36, 94 GHz |

**Figure:** Apparent range rate [m/s] vs carrier frequency [GHz] with
unambiguous bounds.

---

## 3 — Waveforms

### Exercise 3.1: Waveform Cross-Correlations

Generate uncoded, Barker-7, random-7, and LFM pulses.  For each waveform,
plot the time-domain shape, power spectral density, and autocorrelation.

| Parameter   | Value                 |
|-------------|-----------------------|
| Sample rate | 10 (normalised units) |
| Bandwidth   | 1 (normalised units)  |
| Barker code | 7 chips               |
| Random code | 7 chips               |
| LFM pulse   | $T = 2$, up-chirp     |

**Figures:** 8 plots (2 per waveform): pulse shape + PSD, and pulse shape +
autocorrelation.

### Exercise 3.2: Barker Autocorrelation Sidelobes

Overlay the autocorrelation of all Barker code lengths and verify that
sidelobes are at most $1/N$ of the peak.

| Parameter   | Value    |
|-------------|----------|
| Bandwidth   | 4 MHz    |
| Sample rate | 16 MHz   |
| Zero-pad    | 50 samples |

**Figure:** 2 subplots — time-domain pulses overlaid, and autocorrelation
magnitudes overlaid, for uncoded and all Barker lengths (2, 3, 4, 5, 7, 11, 13).

### Exercise 3.3: Noisy Cross-Correlations

Demonstrate matched-filter detection in noise at varying SNRs and show
waveform selectivity.

| Parameter   | Value    |
|-------------|----------|
| Sample rate | 20 MHz   |
| Bandwidth   | 4 MHz    |

**Case 1:** Single uncoded pulse at sample index 200, SNR = 20 dB, 1000
samples of unit-variance noise.

**Figure:** 3 subplots — noise, pulse, matched-filter output.

**Case 2:** Three uncoded pulses at indices 128 (15 dB), 200 (30 dB),
950 (20 dB).

**Figure:** 4 subplots — noise, pulses, matched-filter magnitude, and
matched-filter in dB.

**Case 3:** LFM pulse at index 300 (SNR = 20 dB, $T = 2\ \mu$s, up-chirp)
and Barker-13 at index 600 (SNR = 20 dB).  Apply each matched filter and
show selectivity.

**Figure:** 5 subplots — signal+noise, LFM pulse, BPSK pulse, LFM
matched-filter output, BPSK matched-filter output.

### Exercise 3.4: Barker-13 vs Uncoded — Amplitude and Width

Compare processing gain and mainlobe width between Barker-13 and a single
uncoded chip.

| Parameter   | Value    |
|-------------|----------|
| Bandwidth   | 4 MHz    |
| Sample rate | 16 MHz   |
| SNR         | 20 dB    |

**Figure:** 2 subplots — time-domain pulses, and matched-filter outputs.

### Exercise 3.5: Ambiguity Function

Compute and display the ambiguity function for uncoded, Barker-13, and LFM
waveforms.

| Parameter   | Value    |
|-------------|----------|
| Sample rate | 100 kHz  |
| Bandwidth   | 10 kHz   |
| Barker code | 13 chips |
| LFM pulse   | $T = 1$ ms, up-chirp |

**Figures:** 6 plots (2 per waveform): ambiguity surface (delay vs Doppler
colourmap in dB) and zero-delay / zero-Doppler cuts.

---

## 4 — Datacube Processing

### Exercise 4.1: Datacube Processing Test

Build a raw datacube, inject a tone at a known range bin and Doppler
frequency, and verify it appears in the correct range-Doppler cell.

| Parameter   | Value    |
|-------------|----------|
| Sample rate | 20 MHz   |
| PRF         | 100 kHz  |
| Pulses      | 256      |
| Tone range bin | 98    |
| Tone Doppler   | PRF/4 |

**Figure:** 2 subplots — raw datacube (fast time vs slow time) and processed
range-Doppler map.

### Exercise 4.4: Windowing Comparison

Apply Chebyshev, Blackman-Harris, and Taylor windows to a rectangular pulse
and compare spectral mainlobe and sidelobe levels.

| Parameter   | Value    |
|-------------|----------|
| Sample rate | 100 MHz  |
| Bandwidth   | 11 MHz   |
| Chebyshev   | 60 dB    |
| Taylor      | defaults |
| Zero-pad    | 4001 samples |

**Figures:** 4 plots (one per window including unfiltered): pulse shape and
PSD side by side.

### Exercise 4.5: Complex Tone Windowing

Show spectral leakage suppression when applying a Chebyshev window to a
complex sinusoid.

| Parameter      | Value    |
|----------------|----------|
| Sample rate    | 200 MHz  |
| Bandwidth      | 11 MHz   |
| Tone frequency | $f_s/8$ = 25 MHz |
| Chebyshev      | 60 dB    |

**Figures:** 2 plots — unwindowed tone (pulse + PSD), and Chebyshev-windowed
tone (pulse + PSD).

### Exercise 4.6: Range and Range-Rate from Positions

Compute range and range-rate from 3-D position and velocity vectors.

| Parameter           | Value           |
|---------------------|-----------------|
| Radar position      | [0, 0, 3048] m |
| Radar velocity      | [300, 0, 0] m/s |
| Target position     | [5000, 0, 3048] m |
| Target velocity     | [$-300$, 0, 0] m/s |

**Output:** Printed range vector, range magnitude, and range-rate (no figure).

---

## 5 — Range-Doppler Maps

### Exercise 5.0: Stationary Target RDM

Generate an RDM for a single stationary target and verify the SNR matches the
range equation prediction.

| Parameter          | Value          |
|--------------------|----------------|
| Bandwidth          | 10 MHz         |
| Carrier frequency  | 10 GHz         |
| Transmit power     | 1 kW           |
| Tx/Rx gain         | 30 dB each     |
| Temperature        | 290 K          |
| Sample rate        | 20 MHz         |
| Noise factor       | 8 dB           |
| System losses      | 8 dB           |
| PRF                | 200 kHz        |
| Dwell time         | 2 ms           |
| Waveform           | LFM, $T = 1\ \mu$s, up-chirp |
| Target             | 3.5 km, 0 m/s, 10 dBsm |

**Figure:** Range-Doppler map with SNR annotations.

### Exercise 5.1: CFAR Detection

Apply CA-CFAR, GOCA-CFAR, and SOCA-CFAR to an RDM with three targets.

| Parameter          | Value          |
|--------------------|----------------|
| Radar              | same as 5.0    |
| Waveform           | LFM, $T = 1\ \mu$s, up-chirp |
| Target 1           | 3 km, 0 m/s, 10 dBsm |
| Target 2           | 5 km, $-500$ m/s, 0 dBsm |
| Target 3           | 4 km, 1 km/s, 20 dBsm |
| CA-CFAR Pfa        | $10^{-5}$     |
| Variant comparison Pfa | $10^{-6}$ |
| Guard cells (R/D)  | 3 / 3          |
| Training cells (R/D) | 10 / 10     |

**Figure 1:** CA-CFAR detections overlaid on RDM.
**Figure 2:** 3 subplots comparing CA, GOCA, and SOCA detection counts.

---

## 7 — Linear Arrays

### Exercise 7.0: ULA Array Factor Studies

**Study 1 — Constant aperture length ($L = 5\lambda$):**

| Configuration | Elements | Spacing      |
|---------------|----------|--------------|
| A             | 10       | $\lambda/2$  |
| B             | 20       | $\lambda/4$  |
| C             | 40       | $\lambda/8$  |

**Figure:** Gain [dBi] vs angle overlaying three configurations.

**Study 2 — Constant element spacing ($d_x = \lambda/2$):**

| Configuration | Elements | Aperture      |
|---------------|----------|---------------|
| A             | 4        | $2\lambda$    |
| B             | 8        | $4\lambda$    |
| C             | 16       | $8\lambda$    |

**Figure:** Gain [dBi] vs angle overlaying three configurations.

### Exercise 7.1–7.3: Textbook Figures 24–28

**Figure 24 — Element count comparison:** 10 and 40 elements, $\lambda/2$
spacing.  Two subplots.

**Figure 25 — Element spacing comparison:** 10 elements with $d_x = \lambda/4$,
$\lambda/2$, $\lambda$.  Three subplots showing grating lobe at $d_x = \lambda$.

**Figure 27 — Weighting comparison:** 40 elements, $\lambda/2$ spacing.

| Weight     | Parameter  |
|------------|------------|
| Uniform    | —          |
| Chebyshev  | 30 dB      |
| Taylor     | SLL = 35 dB |

Two subplots: full angular range and zoomed ($\pm 8$°).

**Figure 28 — Beam steering:** 20 elements, $\lambda/2$ spacing.  Steered to
15°, 45°, $-60$°.  Three subplots.

### Exercise 7.5: Monopulse Angle Estimation vs SNR

Estimate target angle with a two-element amplitude monopulse over a range of
SNR values.

| Parameter         | Value               |
|-------------------|---------------------|
| Samples           | 1000                |
| Signal frequency  | 1 Hz                |
| Target angle      | 2°                  |
| Array positions   | $[-\lambda/4, +\lambda/4]$ |
| SNR sweep         | 0 to 27 dB, step 3 |

**Figure 1:** Noisy received signal vs time for each SNR.
**Figure 2:** 2 subplots — mean angle error vs SNR, std dev of error vs SNR.

### Exercise 7.5 (freq): Monopulse — Time vs Frequency Domain

Compare four angle estimation methods across SNR.

| Parameter         | Value               |
|-------------------|---------------------|
| Target angle      | $-5$°               |
| SNR sweep         | $-15$ to 45 dB, step 5 |
| Random seed       | 100                 |

Methods: time-domain monopulse ratio, time-domain phase-only,
frequency-domain monopulse ratio, frequency-domain phase-only.

**Figure 1:** Noisy signal vs time.
**Figure 2:** 2 subplots — angle error vs SNR and std dev vs SNR for all four
methods.

### Exercise 7.6: Phase Centre Location

Compute and plot the phase centre of a Chebyshev-weighted 40-element array.

| Parameter         | Value               |
|-------------------|---------------------|
| Elements          | 40                  |
| Chebyshev weight  | 30 dB               |
| Element positions | 2 mm to 8 mm, linear |

**Figure:** Element weights vs position with vertical line at phase centre.

### Exercise 7.7: Sub-Array Sum and Difference Beams

Split a ULA into two halves and form sum ($\Sigma$) and difference ($\Delta$)
beams for monopulse.

| Parameter          | Value      |
|--------------------|------------|
| Carrier frequency  | 10 GHz     |
| Elements           | 20         |
| Spacing            | $\lambda/2$ |

**Figure:** $\Sigma$ and $\Delta$ gain patterns vs angle ($\pm 20$°).

### Exercise 7.8: Monopulse on Range-Doppler Map

Apply monopulse angle estimation to the peak cell of an RDM generated with a
two-element array.

| Parameter          | Value          |
|--------------------|----------------|
| Radar              | same as 5.0 except PRF = 50 kHz |
| Array spacing      | $\lambda/2$   |
| Target angle       | 5°             |
| Target             | 2.4 km, 200 m/s, 10 dBsm |

**Output:** RDMs for each element; printed monopulse angle estimate and error.

---

## 8 — Synthetic Aperture Radar

### Exercise 8.1: Stripmap SAR Validation

Generate a stripmap SAR image and validate the cross-range resolution against
theory.

| Parameter          | Value          |
|--------------------|----------------|
| Bandwidth          | 5 MHz          |
| Carrier frequency  | 10 GHz         |
| Waveform           | LFM, $T = 10\ \mu$s, up-chirp |
| Tx/Rx gain         | 30 dB each     |
| Temperature        | 290 K          |
| Noise factor       | 8 dB           |
| System losses      | 8 dB           |
| PRF                | 8 kHz          |
| Platform velocity  | 100 m/s        |
| Aperture length    | 50 m           |
| Platform altitude  | 5 km           |
| Target 1           | [$-5$, 100, 0] m, 10 dBsm |
| Target 2           | [0, 3000, 0] m, 10 dBsm |
| Target 3           | [5, 5000, 0] m, 10 dBsm |
| Window             | none (unwindowed) |

**Figure 1:** Full focused SAR image.
**Figure 2:** 3 subplots — zoomed PSF around each target ($\pm 150$ m range,
$\pm 15$ m cross-range).
**Figure 3:** 3 subplots — cross-range cuts at peak range bin with $-3$ dB
width and theoretical resolution annotated.

### Exercise 8.2: Stripmap SAR Peak Analysis

Analyse peak locations and azimuth drift across range bins (same setup as 8.1).

**Output:** Printed peak coordinates and azimuth peak position vs range bin
offset for each target.

### Exercise 8.3: Stripmap vs Spotlight SAR

Compare stripmap and spotlight modes side by side.

| Parameter (common)   | Value          |
|----------------------|----------------|
| All RF parameters    | same as 8.1    |
| Targets              | same as 8.1    |

| Mode       | Aperture Length | Scene Centre       | Beamwidth |
|------------|----------------|--------------------|-----------|
| Stripmap   | 50 m           | —                  | —         |
| Spotlight  | 200 m          | [0, 3000, 0] m    | 0.06 rad  |

**Figure 1:** Stripmap zoomed image ($\pm 15$ m cross-range).
**Figure 2:** Spotlight zoomed image ($\pm 15$ m cross-range).
**Figure 3:** Stripmap cross-range cut with $-3$ dB width annotation.
**Figure 4:** Spotlight cross-range cut with $-3$ dB width annotation.

---

## 9 — Space-Time Adaptive Processing

### Exercise 9.0: STAP Demonstration

Simulate an airborne ULA radar with angle-Doppler coupled clutter and compare
conventional beamforming against STAP (SMI).

| Parameter          | Value           |
|--------------------|-----------------|
| Bandwidth          | 1 MHz           |
| Carrier frequency  | 10 GHz          |
| Waveform           | LFM, $T = 10\ \mu$s, up-chirp |
| Transmit power     | 1 kW            |
| Tx/Rx gain         | 20 dB each      |
| Temperature        | 290 K           |
| Noise factor       | 5 dB            |
| System losses      | 3 dB            |
| PRF                | 10 kHz          |
| Dwell time         | 1.6 ms (16 pulses) |
| Array              | 8 elements, $\lambda/2$ spacing |
| Platform velocity  | 100 m/s         |
| CNR                | 40 (per patch)  |
| Clutter patches    | 180             |
| Steer angle        | 0°              |
| Guard bins         | 3               |
| Diagonal load      | $10^{-2}$       |
| Target 1           | 7 km, $-50$ m/s, 20 dBsm, 20° |
| Target 2           | 10 km, 30 m/s, 10 dBsm, $-10$° |

**Figure:** Side-by-side RDMs — conventional (beamform + FFT) and STAP (SMI).

### Exercise 9.1: STAP Validation

Verify STAP correctness with a single strong target.

| Parameter          | Value           |
|--------------------|-----------------|
| Radar / array      | same as 9.0     |
| Target             | 7 km, $-30$ m/s, 30 dBsm, 0° |

**Test 1 (no clutter):** Verify target peak appears at the correct range and
range-rate bins.  Tolerance: 2 range bins and 2 Doppler bins.

**Test 2 (with clutter):** Measure clutter power in training bins for
conventional vs adaptive.  Expect $> 5$ dB suppression.

**Test 3 (steering vector):** Verify that the spatial component of the
space-time steering vector matches `steering_vector(el_pos, angle)` and the
temporal component matches the Doppler phase ramp.

**Output:** Printed PASS/FAIL for each test.
