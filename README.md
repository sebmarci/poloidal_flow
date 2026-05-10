# W7-X ABES - TDE Poloidal Flow Analysis

This package uses the method of Time Delay Estimation to calculate poloidal flow velocities based on fast modulated Wendelstein 7-X Alkali Beam Emission Spectroscopy measurements.

## What it does

1. Pulls raw ABES signals and chopper timings from W7-X via FLAP.
2. Performs beam-on/beam-off background subtraction per deflection state.
3. Applies an optional bandpass filter.
4. Computes the normalized CCF of `defl0` and `defl1` on a sliding time window.
5. Extracts the CCF peak time lag with one of three fitters
   (`gaussian`, `cubic_spline`, `parabola`) — the parabola fit also
   propagates a 1σ uncertainty.
6. Converts time lag + CMOS-measured poloidal beam separation to a velocity.

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
    channels=list(range(1, 21)),
    bandpass_type='Butterworth',
    bandpass_range=(2e3, 10e3),
)
corr_config = CorrelationConfig(
    xcorr_fitting_method='parabola',
    xcorr_window=0.05,
    xcorr_time_lag_interval=(-1e-4, 1.2e-4),
)

# 2. Read and preprocess both deflection states
defl0, defl1 = ABESDataReader(abes_config).read_data()

# 3. Cross-correlate over a time grid
analyzer = CorrelationAnalysis(defl0, defl1, corr_config)
times = np.linspace(0.5, 9.5, 91)
tau, tau_err, corr = analyzer.get_max_time_lag(times, abes_config.channels)
# tau, tau_err, corr have shape (len(times), len(channels)) — tau in microseconds

# 4. Inspect a single CCF
CCFPlotter(analyzer).plot_single(time=7.0, channel=10, method='parabola')
```

## Package layout

| Module                                | Contents                                                   |
|---------------------------------------|------------------------------------------------------------|
| `poloidal_flow.core.config`           | `ABESConfig`, `CorrelationConfig` dataclasses              |
| `poloidal_flow.core.reading`          | `ABESDataReader` — load, background-subtract, filter       |
| `poloidal_flow.core.analysis`         | `CorrelationAnalysis` — CCF + peak fitting                 |
| `poloidal_flow.visualization.plotting`| `CCFPlotter` — single- and multi-channel CCF figures       |
| `poloidal_flow.beam_axis.pipeline`    | `CVPipeline` — CMOS beam-axis extraction (OpenCV + Huber)  |

Analysis notebooks live in [notebooks/](notebooks/); one-off scripts live in
[scripts/](scripts/).

## License

MIT — see [LICENSE](LICENSE).
