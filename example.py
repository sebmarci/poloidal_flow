# Example script demonstrating the usage of poloidal_flow classes.

import sys

# NOTE: flap_w7x_abes needs to be present in you PYTHONPATH, so you need to change this path appropriately.
# If you have already included it (e.g. via Spyder), you may delete this line.
sys.path.append('/home/smarci/python_libs')

import flap
from poloidal_flow import ABESConfig, ABESDataReader, CorrelationConfig, CorrelationAnalysis
import numpy as np
import matplotlib.pyplot as plt

# Configuration classes for data reading and cross-correlation computation ---

channels = np.arange(1, 21)

abes_config = ABESConfig(
    exp_id = '20250409.046',
    time_range = None,
    channels = channels,
    bandpass_type = 'Butterworth',
    bandpass_range = [2e3, 10e3]
)

corr_config = CorrelationConfig(
    xcorr_fitting_method = 'cubic_spline',
    xcorr_window = 0.05,
    xcorr_time_lag_interval = [-1e-4, 1.2e-4],
)

# Read ABES data in 0 and 1 deflection state (background-subtracted automatically) 

defl0, defl1 = ABESDataReader(abes_config).read_data()

# Compute CCF time lags for given times and channels

analysis = CorrelationAnalysis(defl0, defl1, corr_config)
time_range = np.linspace(0, 10, 101)
tau, correlation = analysis.get_max_time_lag(time_range, channels)

# Plot results

fig, ax = plt.subplots()

ax.set_title('Test CCF time lags')
ax.set_xticks(channels)
ax.set_xlabel('Channel number')
ax.set_ylabel('Time [s]')

pcm = ax.pcolormesh(channels, time_range, tau, cmap = 'bwr', shading = 'nearest', vmin = -20, vmax = 20)
cbar = fig.colorbar(pcm)
cbar.set_label('Time lag [$\\mu$s]')

plt.savefig('test_ccf.png')
