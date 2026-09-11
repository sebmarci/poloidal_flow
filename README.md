# W7-X ABES — Poloidal Flow Analysis

Extract poloidal velocity profiles from fast modulated Wendelstein 7-X Alkali Beam
Emission Spectroscopy (ABES) measurements by Time Delay Estimation.

The ABES beam is deflected up and down by a pair of capacitor plates, giving two
poloidally separated beam paths (`defl0` and `defl1`) measured by the same 40-channel
APD array. A poloidal bulk flow shifts the two light signals in time relative to one
another; the maximum of their cross-correlation function (CCF) gives that time lag, and
together with the poloidal beam separation it gives the poloidal velocity.

## What it does

1. Pulls raw ABES signals and chopper timings from W7-X via FLAP.
2. Performs beam-on/beam-off background subtraction per deflection state.
3. Applies an optional bandpass filter.
4. Computes the normalized CCF (and, for the elliptical model, the ACFs) of `defl0` and
   `defl1` on a sliding time window, per channel.
5. Extracts the CCF peak lag and its 1σ uncertainty with a weighted parabola fit.
6. Converts the lag plus the CMOS-measured poloidal beam separation to a velocity, in
   either the Taylor or the elliptical model.

## Turbulence models

Which model is appropriate depends on how fast the turbulence decays relative to how
fast it propagates. Both are implemented; see Krämer-Flecken et al. 2025, *Velocity
modulations in view of the elliptical approach at Wendelstein 7-X*, Plasma Phys. Control.
Fusion **67** 055024, equations (1)–(3).

**Taylor (frozen turbulence).** The eddies are carried by the flow without changing
shape, so the space-time diagram is a set of straight lines:

```
v_pol = s / dt
```

with `s` the poloidal separation of the two deflected beams and `dt` the CCF maximum lag.

**Elliptical approach (EA).** The turbulence is not frozen: it also decays with a
characteristic fading time, which distorts the space-time contours into ellipses. This
introduces a second timescale `tau_0`, the lag at which the **mean ACF of the two
signals** equals the **CCF maximum value**:

```
v_pol  = s * dt / (dt^2 + tau_0^2)
v_fade = s / sqrt(dt^2 + tau_0^2)
```

Taylor is the `tau_0 -> 0` limit of the EA, and is adequate when `dt / tau_0 >= 5`; below
that it overestimates the velocity. W7-X sits in the regime where `dt` and `tau_0` are
comparable, so the EA is the model of interest here.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.8 and the following on `PYTHONPATH`:

- [`flap`](https://github.com/fusion-flap/flap)
- `flap_w7x_abes` (W7-X ABES data reader for FLAP)

Runtime dependencies: `numpy`, `scipy`, `matplotlib`, `h5py`, `opencv-python`.

## Quick start

```python
import numpy as np
from poloidal_flow import (
    ABESConfig, CorrelationConfig,
    ABESDataReader, CorrelationAnalysis, CCFPlotter,
)

# 1. Configure the shot and the analysis
abes_config = ABESConfig(
    exp_id='20250409.046',
    time_range=(0, 10),               # seconds; None = whole shot
    bandpass_type='Butterworth',      # 'Elliptic', 'Butterworth', or None
    bandpass_range=(2e3, 10e3),       # Hz
)
corr_config = CorrelationConfig(
    xcorr_fitting_method='parabola',          # the only method usable for time lags
    xcorr_window=1.0,                         # CCF time window, seconds
    xcorr_interval=150,                       # CCF subintervals — MUST be > 1
    xcorr_time_lag_interval=(-1e-4, 1.2e-4),  # lag range to keep, seconds
)

# 2. Read and preprocess both deflection states
defl0, defl1 = ABESDataReader(abes_config).read_data()

# 3. Correlate over a grid of time windows and channels
analyzer = CorrelationAnalysis(defl0, defl1, corr_config)
times = np.linspace(0, 10, 101)
channels = np.arange(1, 41)

# Elliptical approach: convective and fading velocity together
tau, tau_err, tau0, corr = analyzer.get_max_time_lag_elliptical(times, channels)
vpol, vpol_err, vfade, vfade_err = analyzer.get_velocity_elliptical(tau, tau_err, tau0)

# Taylor model, for comparison
# tau, tau_err, corr = analyzer.get_max_time_lag_taylor(times, channels)
# vpol, vpol_err = analyzer.get_velocity_taylor(tau, tau_err)

# All arrays have shape (len(times), len(channels)):
#   tau, tau_err, tau0 in microseconds; velocities in km/s (mm/us)

# 4. Inspect a single CCF
CCFPlotter(analyzer).plot_single(time=7.0, channel=20, method='parabola')
```

The beam separation `s` comes from the APDCAM XML configuration of the shot and is read
lazily by the `get_velocity_*` methods:

```python
analyzer.get_deflection_params()
analyzer.deflection_voltage       # (top - bottom) chopper plate voltage, V
analyzer.poloidal_deflection      # separation s, mm
analyzer.poloidal_deflection_err
```

`CALIBRATION_FACTOR` / `CALIBRATION_FACTOR_ERR` (mm/V), measured with `CVPipeline` from
CMOS images of the deflected beam, are module-level constants in `analysis.py` and are
mirrored onto each analyzer instance.

## Repository layout

```
src/poloidal_flow/
├── __init__.py                        # Public API
├── config.py                          # ABESConfig, CorrelationConfig
├── reading.py                         # ABESDataReader
├── analysis.py                        # CorrelationAnalysis
├── plotting.py                        # CCFPlotter
├── pipeline.py                        # CVPipeline (beam axis from CMOS images)
└── poldef_spatcal/
    └── cmos_spatcal_read.py           # CMOS pixel → device coordinate calibration

scripts/
├── process_ccf.py                     # Main driver: time lags + velocities → HDF5
├── save_data.py                       # Read and pickle preprocessed shot data
├── generate_plots_bulk.py             # Plot the HDF5 results of many shots
├── get_apsd.py, get_cpsd.py           # Auto/cross power spectral densities
├── get_cmos_pics.py                   # Collect CMOS beam images for a shot list
├── process_cmos_images.py             # Run CVPipeline over those images
├── poloidal_modulation.py             # Beam deflection modulation diagnostics
└── read_spatcal.py                    # Inspect the ABES spatial calibration

notebooks/                             # Analysis notebooks using the package API
example.py                             # Minimal end-to-end example
```

Not tracked: `flap_defaults.cfg` (FLAP configuration, required at runtime),
`pickled_shot_data/` (preprocessed FLAP `DataObject` pickles),
`processed_data/<exp_id>/*.h5` (time lags and velocities), `cmos/` (beam images),
`plots/`.

| Module | Class | Responsibility |
|--------|-------|----------------|
| `poloidal_flow.config` | `ABESConfig`, `CorrelationConfig` | Acquisition and analysis parameters |
| `poloidal_flow.reading` | `ABESDataReader` | Load, background-subtract, filter |
| `poloidal_flow.analysis` | `CorrelationAnalysis` | CCF/ACF, peak fitting, Taylor + EA velocities |
| `poloidal_flow.plotting` | `CCFPlotter` | Single- and multi-channel CCF figures |
| `poloidal_flow.pipeline` | `CVPipeline` | CMOS beam-axis extraction (OpenCV + Huber fit) |

## Pitfalls

These fail silently — no exception, just wrong numbers.

- **`xcorr_interval` must be > 1.** FLAP derives the CCF error from the scatter across
  subintervals, so with one subinterval the errors are all zero and the weighted parabola
  fit returns its initial guess unchanged (`tau = 1.0 µs`, `tau_err = nan`) for every
  window. The `CorrelationConfig` default of `1` is a trap; pass 100–150 explicitly.
- **`xcorr_normalize` must stay True for the elliptical model.** `ACF(tau_0) = CCF_max` is
  only meaningful if both are normalized to 1 at zero lag.
- **Only `'parabola'` yields time lags.** `fit_gaussian` and `fit_cubic_spline` are
  deprecated and return 3-tuples, which breaks `get_max_time_lag_*`. They survive only so
  that `CCFPlotter` can still draw them.
- **CCF argument order sets the velocity sign.** Both `get_max_time_lag_*` methods call
  `ccf_window_single(defl1_single, defl0_single)`; swapping the arguments flips the sign of
  `dt` and hence of `v_pol`.
- **Edge peaks are not fitted.** A CCF peak within 2 samples of the edge of
  `xcorr_time_lag_interval` means the true maximum lies outside it, so the fit is skipped
  and `nan`s are returned. Set the lag interval to match the chopping and deflection
  frequency of the beam.

The elliptical model computes two ACFs on top of the CCF per (time, channel) point, so it
costs roughly 2–3× the Taylor model.

## Known limitations

- `tau_0` has no uncertainty estimate, so neither elliptical error bar is complete. For
  `v_pol` the missing term is the dominant one, so `vpol_err` is a zero placeholder;
  `vfade_err` propagates the `s` and `dt` terms and is a lower bound.
- The `tau_0` root find is an unbracketed `fsolve` from a hardcoded 20 µs initial guess.
  Non-convergence is detected and raises a `RuntimeError` (which aborts the whole run,
  including on a `nan` `corr` from a failed peak fit), but a converged solve can still land
  on a spurious root in the spline's extrapolation region.
- `CCFPlotter` calls `ccf_window_single(defl0, defl1)`, the reverse of the order used by
  the analysis methods, so plotted lags have the opposite sign to analysed ones. It also
  skips `truncate_data`.
- `example.py` and several notebooks still call the removed `get_max_time_lag()` /
  `get_poloidal_deflection_voltage()` API and no longer run.
- `scripts/process_cmos_images.py` looks up the spatial calibration under a path from an
  older nested layout (`src/poloidal_flow/beam_axis/poldef_spatcal/`); it now lives in
  `src/poloidal_flow/poldef_spatcal/`.
- Everything below the deprecation marker in `analysis.py` (radial correlation, 2D CCF) is
  kept for reference only.

## License

MIT — see [LICENSE](LICENSE).
