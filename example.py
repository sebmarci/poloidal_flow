# Example script demonstrating the usage of poloidal_flow classes.

import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp

from poloidal_flow import ABESConfig, ABESDataReader, CorrelationConfig, CorrelationAnalysis

# Configuration classes for data reading and cross-correlation computation ---

abes_config = ABESConfig(
    exp_id = '20250409.046',
    time_range = None,
    bandpass_type = 'Butterworth',
    bandpass_range = [2e3, 10e3]
)

corr_config = CorrelationConfig(
    xcorr_fitting_method = 'parabola',  # the only method usable for time lags
    xcorr_window = 0.05,
    xcorr_interval = 150,               # CCF subintervals - MUST be > 1
    xcorr_time_lag_interval = [-1e-4, 1.2e-4],
)

# Read ABES data in 0 and 1 deflection state (background-subtracted automatically)

defl0, defl1 = ABESDataReader(abes_config).read_data()

# Compute CCF time lags for given times and channels (Taylor model)

analysis = CorrelationAnalysis(defl0, defl1, corr_config)
time_range = np.linspace(0, 10, 101)
channels = np.arange(1, 21)
tau, corr = analysis.get_max_time_lag_taylor(time_range, channels)

# Plot results. `tau` is an object array of `uncertainties.ufloat`, so the
# nominal values have to be pulled out before handing them to matplotlib.

fig, ax = plt.subplots()

ax.set_title('Test CCF time lags')
ax.set_xticks(channels)
ax.set_xlabel('Channel number')
ax.set_ylabel('Time [s]')

pcm = ax.pcolormesh(
    channels, time_range, unp.nominal_values(tau),
    cmap = 'bwr', shading = 'nearest', vmin = -20, vmax = 20
)
cbar = fig.colorbar(pcm)
cbar.set_label('Time lag [$\\mu$s]')

plt.savefig('test_ccf.png')
