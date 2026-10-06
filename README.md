# fmcw-radar-sim

Simulation of an FMCW radar signal processing chain in Python: LFM chirp generation, echo and noise model, dechirping, anti-aliasing filter and ADC, range–Doppler processing, 2D CA-CFAR detection, range/velocity estimation and a Monte Carlo evaluation of the detector.

Revised version of the radar signal detector term project of a Digital Signal Processing course.

## Background

This project is an improved version of a pulsed LFM radar simulation done as a Digital Signal Processing term project. The earlier version used a 10 GHz pulsed LFM waveform with matched filtering, a Doppler FFT over 16 pulses, 1D CA-CFAR along the Doppler axis and a Monte Carlo curve of detection probability against SNR, with a single target at 4.5 km.

Following the feedback that 4.5 km is not an automotive distance, this version moves to the waveform of automotive radars: a 77 GHz FMCW chirp with dechirping, a maximum range of 160 m by design and six targets between 18 and 150 m. A strong reflector at 470 m, outside the designed range, shows the effect of the anti-aliasing filter.

|                      | Older version             | Newer version                  |
| -------------------- | ------------------------- | ------------------------------ |
| Waveform             | pulsed LFM, 10 GHz        | FMCW, 77 GHz                   |
| Bandwidth            | 20 MHz                    | 150 MHz                        |
| Range resolution     | 7.5 m                     | 1 m                            |
| Range processing     | matched filter            | dechirp, range FFT             |
| Slow time            | 16 pulses, 200 µs         | 128 chirps, 30 µs              |
| Velocity resolution  | 4.7 m/s                   | 0.51 m/s                       |
| Scene                | 1 target at 4.5 km        | 6 targets at 18–150 m          |
| CFAR                 | 1D, along Doppler         | 2D, range × Doppler            |
| Window before CFAR   | none                      | Hann                           |
| Evaluation           | P_d vs SNR                | P_d vs SNR, measured P_fa      |

The older version kept the Doppler FFT unwindowed before CFAR to keep the cells independent. The Monte Carlo study in this version shows the effect of windowing on the false-alarm rate (see below).

## Processing chain

1. Complex baseband LFM chirp, 128 chirps per frame.
2. Echo for each point target with a time-varying delay, so range and Doppler come from the same model.
3. Complex white Gaussian noise, scaled to a fixed per-sample SNR at the ADC output.
4. Dechirping (`tx · conj(rx)`), Kaiser-window FIR low-pass filter, decimation from 200 MHz to 20 MHz complex sampling.
5. Range FFT along fast time and Doppler FFT along slow time, both Hann-windowed.
6. 2D cell-averaging CFAR on the range–Doppler power map.
7. Clustering of detected cells, parabolic sub-bin interpolation, range–Doppler coupling correction.
8. Monte Carlo evaluation of the false-alarm rate and the detection probability.

## Radar parameters

| Parameter                     | Value                         |
| ----------------------------- | ----------------------------- |
| Carrier frequency             | 77 GHz (λ = 3.9 mm)           |
| Chirp slope                   | 7.5 MHz/µs                    |
| Ramp / idle time              | 24 µs / 6 µs                  |
| Chirp repetition interval     | 30 µs                         |
| ADC window                    | 2–22 µs after ramp start      |
| Sampling rate (complex)       | 20 MHz, 400 samples per chirp |
| Chirps per frame              | 128 (3.84 ms)                 |
| Range resolution              | 1.0 m (150 MHz in ADC window) |
| Maximum range                 | 160 m (set by the filter)     |
| Velocity resolution           | 0.51 m/s                      |
| Maximum unambiguous velocity  | ±32.5 m/s                     |

## Signal model

```
s(t)     = exp[ j2π (f0 t + S t²/2) ],          0 <= t <= T_ramp
tau_k(t) = 2 (R_k + v_k t) / c
r(t)     = sum_k A_k s(t - tau_k) exp(-j2π fc tau_k + j phi_k) + w(t)
A_k      = A_ref sqrt(sigma_k) (100 m / R_k)^2
w(t)     ~ CN(0, sigma_w^2),   sigma_w^2 = 1 / sum(h[n]^2)
```

`t` runs over both fast and slow time, so the Doppler shift follows from the target motion and is not added as a separate term. `A_ref` gives a per-sample SNR of −15 dB for a 1 m² target at 100 m. The noise scaling makes the noise power 1 after the anti-aliasing filter `h[n]`.

After dechirping, each target gives a tone at `f_b = S tau = 2SR/c`. Between chirps the delay changes by `2 v T_c / c`, which turns the carrier phase by `4π v T_c / λ`; the Doppler FFT measures this phase step. The parameters set the following quantities:

| Quantity                     | Relation          | Value      |
| ---------------------------- | ----------------- | ---------- |
| Range resolution             | c / 2B            | 1.0 m      |
| Maximum range                | c f_cut / 2S      | 160 m      |
| Maximum velocity             | ± λ / 4T_c        | ±32.5 m/s  |
| Velocity resolution          | λ / (2 N_c T_c)   | 0.51 m/s   |

The beat frequency also contains a Doppler term `2v/λ` of at most 16.7 kHz (0.33 m in range). It is removed after Doppler estimation with `R = c (f_b - f_D) / 2S`.

## Detection

2D CA-CFAR with a square-law detector, 2 guard cells per side in range and Doppler, 8 training cells per side in range and 4 in Doppler (N = 248), and P_fa = 10⁻⁶:

```
alpha = N (P_fa^(-1/N) - 1) = 14.2   (11.5 dB)
```

The Doppler axis is wrapped (the FFT output is periodic); the range axis is mirrored at the edges. The noise estimate for all cells is one 2D convolution with a ring-shaped kernel.

## Usage

Python 3.10 or later.

```
pip install -r requirements.txt
python fmcw_radar_sim.py
python make_report_figures.py
python monte_carlo.py              # about 3 minutes on two cores
python monte_carlo.py --plot-only  # figures from the saved results
```

## Outputs

- Console table of true and estimated range, velocity and SNR, with and without the anti-aliasing filter
- `figures/1_beat_spectrum.png` — spectrum of one dechirped chirp and the filter response
- `figures/2_range_profile.png` — range profile of a single chirp
- `figures/3_range_doppler.png` — range–Doppler maps with and without the filter
- `figures/4_cfar_cut.png` — cell power and CFAR threshold along range at v = 0
- `results/pfa.csv`, `results/pd.csv`, `results/monte_carlo_log.txt` — Monte Carlo results
- `figures/report/` — figures used in the slides

## Results

One noise realization (seed 7), with the anti-aliasing filter:

| Target | R true [m] | R est. [m] | v true [m/s] | v est. [m/s] | SNR [dB] |
| ------ | ---------- | ---------- | ------------ | ------------ | -------- |
| T1     | 18.0       | 17.97      | −12.0        | −11.99       | 48.3     |
| T2     | 47.0       | 47.01      | 5.0          | 4.99         | 46.0     |
| T3     | 48.5       | 48.50      | −3.0         | −2.99        | 42.9     |
| T4     | 85.0       | 84.99      | 0.0          | 0.01         | 43.6     |
| T5     | 122.0      | 122.05     | 20.0         | 20.00        | 34.6     |
| T6     | 150.0      | 149.95     | −25.0        | −24.99       | 27.1     |

A strong reflector at 470 m lies outside the filter passband and is not detected. Without the filter it aliases to 70 m, +8 m/s and the noise floor rises by 10.2 dB.

## Monte Carlo evaluation

False-alarm rate: 600 noise-only frames through the full chain (1.9 × 10⁷ cells).

| Design P_fa | Measured P_fa | False alarms | i.i.d. reference |
| ----------- | ------------- | ------------ | ---------------- |
| 10⁻²        | 1.13 × 10⁻²   | 219 987      | 1.00 × 10⁻²      |
| 10⁻³        | 1.32 × 10⁻³   | 25 662       | 1.01 × 10⁻³      |
| 10⁻⁴        | 1.66 × 10⁻⁴   | 3 232        | 1.00 × 10⁻⁴      |
| 10⁻⁵        | 2.26 × 10⁻⁵   | 439          | 1.04 × 10⁻⁵      |
| 10⁻⁶        | 3.65 × 10⁻⁶   | 71           | 0.93 × 10⁻⁶      |

The same CFAR applied to i.i.d. exponential cells gives the design value. In the simulated maps, the Hann windows and the zero padding correlate neighboring cells, so the estimate behaves as if it used about 53 independent training cells instead of 248, and the false-alarm rate rises. The threshold multiplier should be set from simulation rather than from the i.i.d. formula.

Detection probability: one target at 100 m, +5 m/s, 150 trials per SNR point, design P_fa = 10⁻⁶. A trial counts as a hit when a detection lies within 3 m and 2 m/s of the target. The processing gain from one ADC sample to the range–Doppler cell is 43.3 dB. P_d = 0.5 at about 11.4 dB in the cell (−32 dB per sample), in agreement with the CA-CFAR curve for a non-fluctuating target; the CFAR loss relative to a known noise level is 0.13 dB.

## Limitations

- Single receive channel, no angle estimation.
- Point targets, no clutter or multipath.
- Ideal hardware: no phase noise, IQ imbalance or ADC quantization.
- Constant velocity over the frame; range migration is below 10 cm.
