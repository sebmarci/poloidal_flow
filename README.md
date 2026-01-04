# Poloidal Flow Analysis for W7-X ABES

A Python package for signal processing and cross-correlation analysis of Alkali Beam Emission Spectroscopy (ABES) data from the Wendelstein 7-X stellarator to calculate poloidal flow velocities.

## Overview

This package provides tools for analyzing ABES diagnostic data from W7-X experiments, specifically designed to measure the poloidal velocity profile through cross-correlation analysis of different ABES beam deflection states.

### Key Features

- **Data Acquisition**: Interface with W7-X ABES diagnostic system via FLAP framework
- **Background Subtraction**: Automatic beam-on/beam-off background removal
- **Signal Filtering**: Apply bandpass filters to select appropriate frequency domains
- **Cross-Correlation Analysis**: Compute maximum CCF time lags between deflection states with multiple fitting methods:
  - Gaussian curve fitting
  - Cubic spline interpolation with peak finding
- **Spatial Calibration**: Support for amplitude and spatial calibration of ABES channels
- **Data Persistence**: Save and load processed data using pickle format

## Installation

**NOTE**: The `flap_w7x_abes` module must be available in your Python path before installing this package.

The package requires:
- Python 3.8+
- [FLAP (Fusion Library of Analysis Programs)](https://github.com/fusion-flap/flap)
- **flap_w7x_abes** - W7-X ABES data reader for FLAP (must be in PYTHONPATH)
- NumPy, SciPy, Matplotlib

