# Wendelstein 7-X ABES Poloidal Flow Analysis

This is a scientific python codebase aimed at processing and analyzing Wendelstein 7-X Alkali Beam Emission Spectroscopy (ABES) data. The main goal of this package is to extract poloidal velocity profiles from modulated ABES measurements. 

## Working principle

Wendelstein 7-X is an experimental stellarator, where an Alkali Beam Emission Spectroscopy (ABES) diagnostic operates. The ABES beam is beam made of neutral alkali atoms (usually sodium), which are accelerated to a certain energy, then injected into the plasma. Due to collisional excitations with plasma particles, the beam will emit a light profile along the beam axis, whose intensity at a certain beam z coordinate depends on the electron density. An avalanche photodiode (APD) system measures the light profile at 40 different positions (channels). The ABES beam can be deflected upwards and downwards using a pair of capacitor plates to which a pre-determined voltage is applied. These two deflection states are named `defl0` and `defl1` in the codebase. If there is a poloidal bulk flow present in the plasma, the defl0 and defl1 signals will be slightly shifted from one another in time. Computing the (normalized) cross-correlation function (CCF) of the two signals, it is possible to determine the poloidal velocity from the maximum CCF time lag, and the poloidal separation between the beams.

This project builds on the FLAP framework, for which the source code is available in `claude_context/flap`. The code uses FLAP to read and store ABES signal data in the form of `DataObject`s. FLAP is also used to compute the cross-correlation function between two signals. It is preferred to store any kind of ABES signal data in `DataObject`s. 

### Turbulence models

Two models are implemented for turning CCF time lags into a velocity. Which one is
appropriate depends on how fast the turbulence decays relative to how fast it propagates.

**Taylor (frozen turbulence).** The eddies are assumed to be carried by the flow without
changing shape, so the space-time diagram is a set of straight lines:

```
v_pol = s / dt
```

where `s` is the poloidal separation of the two deflected beams and `dt` the CCF maximum
time lag.

**Elliptical approach (EA).** The turbulence is *not* frozen: it also decays with a
characteristic fading time, which distorts the space-time contours into ellipses. This
adds a second timescale `tau_0`, the lag at which the **mean ACF of the two signals**
equals the **CCF maximum value**:

```
v_pol  = s * dt / (dt^2 + tau_0^2)
v_fade = s / sqrt(dt^2 + tau_0^2)
```

Taylor is the `tau_0 -> 0` limit of the EA. The rule of thumb from the reference paper is
that Taylor is adequate when `dt / tau_0 >= 5`; below that the EA is required or the
velocity is overestimated. W7-X sits in the regime where the EA matters (`dt` and `tau_0`
are comparable), which is why the EA is the model of interest here.

Reference paper (in `claude_context/`): Krämer-Flecken et al. 2025, *Velocity modulations
in view of the elliptical approach at Wendelstein 7-X*, Plasma Phys. Control. Fusion 67
055024. Equations (1)-(3) are the three formulas above; equation (4) gives the closed form
`tau_0 = sigma_ACF * sqrt(ln(1/CCF_max) / ln 2)` for a Gaussian ACF, useful as a sanity
check (note the paper prints it as `ln(CCF)`, which is a typo — that argument is negative).

## Package structure

The layout is flat; there are no `core/`, `beam_axis/` or `visualization/` subpackages.

```
src/poloidal_flow/
├── __init__.py              # Public API: ABESConfig, CorrelationConfig,
│                            #   ABESDataReader, CorrelationAnalysis, CCFPlotter
├── config.py                # Configuration dataclasses
├── reading.py               # Data acquisition and preprocessing
├── analysis.py              # CCF/ACF, peak fitting, Taylor + elliptical models
├── plotting.py              # CCFPlotter
├── pipeline.py              # CVPipeline: beam axis calibration (OpenCV + Huber line fit)
└── poldef_spatcal/cmos_spatcal_read.py

scripts/                     # Standalone analysis / IO scripts
notebooks/                   # Analysis notebooks using the package API
claude_context/flap          # Symlink to the FLAP source (read-only reference)
claude_context/*.pdf         # Reference papers
```

## Design principles

Instance-based classes holding their own state and cached data, configured by dataclasses
passed to the constructor. One responsibility per class. That is the whole architecture.

This project prioritizes **simple, transparent scientific code** over software architecture
complexity. Default to the **simplest solution that is correct, readable, and testable**,
and prefer extending existing patterns over introducing new design styles.

Do **NOT** introduce: generic helper layers, configuration systems for small parameter
sets, premature modularization across many files, abstract base classes, inheritance
hierarchies, plugin or registry systems, factory patterns, or static classes.

## Core components

| Class | Responsibility | Configured by |
|-------|---------------|---------------|
| `ABESConfig` | Data acquisition parameters | is config |
| `CorrelationConfig` | Analysis parameters | is config |
| `ABESDataReader` | Data loading and preprocessing | `ABESConfig` |
| `CorrelationAnalysis` | CCF/ACF, peak fitting, time lags, Taylor + EA velocities | `CorrelationConfig` |
| `CCFPlotter` | Single-channel and multi-channel CCF plots | wraps a `CorrelationAnalysis` |
| `CVPipeline` | CMOS image → beam axis line fit in device coordinates | constructor args |

### Configuration (`config.py`)

```python
from poloidal_flow import ABESConfig, CorrelationConfig

abes_config = ABESConfig(
    exp_id='20250409.046',
    time_range=(0.5, 1.5),            # None = whole shot
    bandpass_type='Butterworth',      # 'Elliptic', 'Butterworth', or None (default)
    bandpass_range=(2e3, 10e3),       # Hz
    spatial_cal=False,                # attach a 'Device R' coordinate
    spatcal_exp_id=None,              # defaults to exp_id
)

corr_config = CorrelationConfig(
    xcorr_fitting_method='parabola',          # the only method usable for time lags
    xcorr_window=1.0,                         # CCF time window, seconds
    xcorr_interval=150,                       # CCF subintervals - MUST be > 1
    xcorr_time_lag_interval=(-1e-4, 1.2e-4),  # lag range to keep, seconds
    xcorr_resolution=None,                    # None = highest available
    xcorr_normalize=True,                     # MUST be True
    apdcam_path='/data2/W7-X/APDCAM',         # XML configs, for the deflection voltage
)
```

⚠️ The `xcorr_interval=1` default is a trap — always pass it explicitly. See
**FLAP / numerical gotchas**.

### Data reading (`reading.py`)

```python
reader = ABESDataReader(abes_config)
data_defl0, data_defl1 = reader.read_data()
device_r = reader.read_spatial_calibration()   # 'Device R' per channel, m
```

- `read_data()` - full pipeline: raw → background subtraction → bandpass
- `read_data_raw()` / `read_timings()` - fetch and cache raw signals / chopper timings
- `background_subtraction(deflection)` - beam-on/off split, interpolate, subtract
- `apply_bandpass(dataobject)` - apply the configured filter

Background subtraction: slice into beam-on and beam-off periods, average the samples in
each chopper period, linearly interpolate the beam-off signal onto the beam-on time points,
subtract, then bandpass if configured.

### Correlation analysis (`analysis.py`)

```python
analyzer = CorrelationAnalysis(data_defl0, data_defl1, corr_config)

times = np.linspace(0, 10, 101)  # seconds
channels = np.arange(1, 41)

tau, corr        = analyzer.get_max_time_lag_taylor(times, channels)
tau, tau0, corr  = analyzer.get_max_time_lag_elliptical(times, channels)

# All three are object arrays of `uncertainties.ufloat`, shape
# (len(times), len(channels)):
#   tau  - CCF maximum time lag, microseconds
#   tau0 - elliptical fading timescale, microseconds (nominal value always positive)
#   corr - CCF value at the peak

# Velocities in km/s (separation in mm / lag in us), also `ufloat` object
# arrays. Both lazily call get_deflection_params() if the deflection has not
# been read yet.
vpol = analyzer.get_velocity_taylor(tau)   # eq. (1)

# The elliptical approach returns the convective velocity (eq. 2) and the
# fading velocity (eq. 3) together - they are two readings of the same fit.
vpol, vfade = analyzer.get_velocity_elliptical(tau, tau0)

# Deflection voltage and beam separation from the APDCAM XML config
analyzer.get_deflection_params()
analyzer.deflection_voltage       # (top - bottom) chopper voltage, V
analyzer.poloidal_deflection      # CALIBRATION_FACTOR * voltage, mm, a ufloat
```

`CALIBRATION_FACTOR` (mm/V) is a module-level `ufloat` constant in `analysis.py`.

Uncertainties are propagated with the `uncertainties` package: `tau`, `corr`,
`poloidal_deflection`/`calibration_factor`, `vpol` and `vfade` are `ufloat`s (or object
arrays of them), and ordinary `+ - * / **` on them propagates the error automatically. The
one exception is `tau_0`, computed manually — see below — because its derivation goes
through an `fsolve` root-find that `uncertainties` cannot differentiate through; only the
final `(value, error)` pair is wrapped into a `ufloat` afterwards so it can be combined
with `tau` automatically in `get_velocity_elliptical`. When writing results to HDF5, split
back into separate nominal/error datasets with `unumpy.nominal_values(x)` /
`unumpy.std_devs(x)` right before `create_dataset` — see `scripts/process_ccf.py`.

Supporting methods: `ccf_window_single(data0, data1)`, `acf_window_single(data)`,
`truncate_data(data0, data1)` (trims both states to a common sample count), and
`fit_parabola(ccf)`.

**Peak fitting.** `fit_parabola` fits `a * (x - x0)^2 + b` to the 5 samples around
`argmax(|CCF|)`, weighted by the CCF errors, and returns `(tau, peak_value, popt)` where
`tau` and `peak_value` are `ufloat`s built from the fit covariance diagonal. The parabola is
parameterized by its extremum, so `peak_value` is identically `b` — needed by the
elliptical `tau_0` error. A peak within 2 samples of the edge of the lag interval means the
true maximum lies outside it, so the fit is skipped and `ufloat(nan, nan)`s are returned.
`fit_gaussian` and `fit_cubic_spline` are deprecated: they return plain floats with no
uncertainty, so selecting them in `CorrelationConfig` silently breaks the error propagation
`get_max_time_lag_*` relies on. They survive only because `CCFPlotter` can still draw them.

**How `get_max_time_lag_elliptical` finds `tau_0`:**
1. Slice both deflection states to the time window, `truncate_data` to a common length.
2. Per channel: compute the ACF of each state, average them (`acfmean`).
3. Compute the CCF and fit the peak → `dt` (`tau`) and the peak value `corr`.
4. Cubic-spline the mean ACF and solve `ACF(tau_0) = corr` with `fsolve`, using
   `corr.nominal_value` since `fsolve` needs plain floats.
5. `tau_0 = abs(root)`, since the ACF is symmetric and only `tau_0^2` is physical.
6. `tau0_err = sqrt(acf_err(tau_0)^2 + corr_err^2) / |ACF'(tau_0)|`, the first order
   propagation of step 4, computed by hand (not by `uncertainties`). Implicit
   differentiation of `ACF(tau_0) - corr = 0` gives both the ACF curve and the crossing
   level the same `1/ACF'(tau_0)` sensitivity, with *opposite* signs. The two are added in
   quadrature, which drops the `-2*Cov(ACF, corr)` cross term — the errors are in fact
   positively correlated (same subintervals, same window), so this is an upper bound.
   Diverges as `tau_0 -> 0`, where the ACF is flat. Full derivation in
   `error_propagation.md`. The result is then wrapped as `ufloat(tau0, tau0_err)`.

⚠️ `fsolve` returns its initial guess (`x0 = 20e-6`) on failure, which is indistinguishable
from a genuine 20 µs root. The solver status `ier` is therefore checked and a `RuntimeError`
is raised on non-convergence. This aborts the whole run rather than skipping one channel —
in particular a `nan` `corr` from a failed/edge parabola fit will raise. Keep the check; if
per-channel tolerance is wanted, short-circuit on `np.isnan(corr.nominal_value)` *before*
the solve instead of weakening the guardrail.

Everything below the `--- THE FOLLOWING FUNCTIONS ARE UNUSED AND/OR DEPRECATED ---` marker
in `analysis.py` (radial correlation, 2D CCF) is kept for reference only. Do not extend it.

### Visualization (`plotting.py`)

```python
plotter = CCFPlotter(analyzer)
fig, ax   = plotter.plot_single(time=7.0, channel=20, method='parabola')
fig, axes = plotter.plot_all_channels(time=7.0, method='parabola', figsize=(40, 40))
```

`method` accepts `'parabola'` (also draws horizontal error bars on the peak), `'gaussian'`
or `'spline'`.

### Beam axis calibration (`pipeline.py`)

`CVPipeline` turns a CMOS image of the beam into a beam axis line in device coordinates:
ROI mask + median blur + Otsu threshold, per-row centroid extraction with width /
pixel-count rejection, pixel → device coordinates via `flap_w7x_abes.ShotSpatCalCMOS`,
then a robust `cv2.fitLine` (Huber) fit. This is what measures the poloidal separation
between the deflection states.

## FLAP / numerical gotchas

These have all bitten this codebase at least once. They fail **silently** — no exception,
just wrong numbers — so they are worth remembering.

- **`xcorr_interval` must be > 1.** FLAP derives the CCF error from the scatter across
  subintervals. With `Interval_n = 1` the error array is all zeros, and
  `curve_fit(..., sigma=zeros)` does *not* raise — it emits an `OptimizeWarning` and
  returns `p0` unchanged, so every window reports `tau = ufloat(1.0, nan)` µs.
  The `CorrelationConfig` default of `1` is a trap; production scripts use 100-150.
- **`xcorr_normalize` must be True for the elliptical model.** `ACF(tau_0) = CCF_max` is
  only meaningful if both are normalized to 1 at zero lag. With `Normalize=False` FLAP
  returns covariances and the equation is meaningless.
- **Always pass `'Verbose': False` in FLAP `ccf` options.** The default is `True` and it
  prints one line per subinterval. At 150 intervals × 101 times × 40 channels × 3
  correlations that is over a million lines of output.
- **For an ACF, call `data.ccf(coordinate=...)` with no reference.** Passing `data` as its
  own reference pushes FLAP down a branch that computes two extra FFT autocorrelations.
  Verified identical output (to 2e-16) at roughly half the runtime.
- **`truncate_data` must update `.shape` as well as `.data`.** FLAP uses `DataObject.shape`
  (not `data.shape`) to generate coordinate arrays, so a stale `.shape` makes
  `coordinate('Time')` return more samples than the data has.
- **CCF argument order sets the velocity sign.** `ccf_window_single(data0, data1)`
  internally computes `data1.ccf(data0)`. Both `get_max_time_lag_*` call it as
  `ccf_window_single(defl1_single, defl0_single)`. Swapping the arguments flips the sign of
  `dt` and hence of `v_pol`; keep the two models consistent with each other.
- **`fsolve` failure is silent.** It returns `x0`. Always use `full_output=True` and check
  `ier`. Note that `ier == 1` still does not guarantee a *useful* root — it can converge to
  a far spurious root because `CubicSpline` extrapolates outside the lag window by default.
- **Guard divisions by `tau`.** `get_velocity_taylor` masks out `|tau| < 1e-3` (via
  `unumpy.nominal_values(tau)`, since a `ufloat` comparison isn't meaningful) before
  dividing, because a near-zero lag otherwise blows the velocity up to infinity.
- **`uncertainties` needs plain floats at FFI boundaries.** `scipy.optimize.fsolve` and
  numpy's own `argmax`/comparisons don't understand `ufloat`s. Extract
  `.nominal_value`/`.std_dev` (or `unumpy.nominal_values`/`unumpy.std_devs` for arrays)
  before handing a value to `fsolve`, `h5py.create_dataset`, or matplotlib.

## Performance

The elliptical model computes two ACFs on top of the CCF for every (time, channel) point,
so it costs roughly 2-3x the Taylor model. The dominant cost is FLAP's correlation calls.

Slice **time first, then channel**. Pre-slicing channels outside the time loop is slower.

## Known issues / work in progress

- `vpol`/`vfade`'s uncertainties are propagated automatically by `uncertainties` from `s`,
  `dt` and `tau_0`, which treats the three as independent `Variable`s. They are not
  entirely: `tau_0` is defined through the CCF maximum value, which comes from the same
  peak fit as `dt`, so a `Cov(dt, tau_0) = pcov[x0, b] / ACF'(tau_0)` cross term is missing
  — `uncertainties` has no way to know `tau` and `tau0` share an origin, since `tau0` is
  constructed as a fresh `ufloat` from the manual `fsolve` result rather than derived by
  arithmetic from `tau`. Consistent with `tau0`'s own uncertainty also dropping its
  covariance; see `error_propagation.md` §8. Capturing this would need
  `uncertainties.correlated_values(popt, pcov)` in `fit_parabola` to keep `tau` and `corr`
  correlated, and then plumbing that correlation through to `tau0` — not currently done.
- The `fsolve` root find for `tau_0` is unbracketed and starts from a hardcoded
  `x0 = 20e-6`. It is guarded (raises on non-convergence) but not robust; a bracketed
  `brentq` on the first sign change of `acfmean - corr` at positive lag would remove both
  the `x0` sensitivity and the risk of converging on a spline-extrapolation root.
- `CCFPlotter` calls `ccf_window_single(defl0, defl1)`, the reverse of the argument order
  used by `get_max_time_lag_*`, so plotted time lags have the opposite sign to analysed
  ones. It also skips `truncate_data`.
- `example.py` and several notebooks still call the old `get_max_time_lag()` /
  `get_poloidal_deflection_voltage()` API and no longer run.
- `scripts/process_cmos_images.py` points its spatcal lookup at
  `src/poloidal_flow/beam_axis/poldef_spatcal/`, a path from the old nested layout. The
  calibration now lives in `src/poloidal_flow/poldef_spatcal/`.
- `calculate_radial_correlation` (deprecated) takes a `time` argument and never uses it.

**When in doubt**: keep it simple. Add complexity only when needed.

# Tool Activation Gates & Context Controls

## 1. Graph & AST Analysis

* *codegraph (MCP)*

  * *Precondition*: Project size exceeds 5,000 lines or task requires tracing multi-file caller/callee trees. Run incremental index (codegraph-mcp index) before querying.

  * *Rule*: Query symbol relations via query_codebase or AST tools instead of dumping whole source files into context.

# 2. High-Token MCP Tools (On-Demand Only)

* *context7 (MCP)*

  * *Precondition*: Require ONLY when fetching specific third-party library documentation or unindexed framework APIs (use context7).
  
  * *Rule*: Always pass target library IDs to query-docs. Discard retrieved doc chunks immediately after code generation is complete.
