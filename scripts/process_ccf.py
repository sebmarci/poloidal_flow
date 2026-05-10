# Calculate CCF time lags and poloidal velocities for one experiment

import sys
import os

sys.path.append('/home/smarci/python_libs')

from poloidal_flow import ABESConfig, ABESDataReader, CorrelationAnalysis, CorrelationConfig
import flap
import flap_w7x_abes
import numpy as np
import h5py

CALIBRATION_FACTOR = 0.09477279 # mm/V
CALIBRATION_FACTOR_ERR = 0.01016466 # mm/V

exp_id = sys.argv[1]
xcorr_fitting_method = 'parabola'
xcorr_window = 0.1 # 100 ms
xcorr_interval = 50
bandpass_range = [2e3, 10e3]
time_range = [0, 10]

output_path = 'processed_data'
output_file = f'{exp_id}_parabola.h5'

config = ABESConfig(
        exp_id = exp_id,
        time_range = time_range,
        bandpass_type = 'Butterworth',
        bandpass_range = bandpass_range
)

xconfig = CorrelationConfig(
    xcorr_fitting_method = xcorr_fitting_method,
    xcorr_window = xcorr_window,
    xcorr_interval = xcorr_interval
)

print('Reading')

defl0, defl1 = ABESDataReader(config).read_data()
anal = CorrelationAnalysis(defl0, defl1, xconfig)

Udefl = anal.get_poloidal_deflection_voltage()
poloidal_separation = CALIBRATION_FACTOR * Udefl
poloidal_separation_err = CALIBRATION_FACTOR_ERR * Udefl

time_range = np.linspace(0, 10, 101)
channels = np.arange(1, 41)

tau, tau_err, corrs = anal.get_max_time_lag(time_range, channels)
vpol = np.divide(poloidal_separation, tau, out = np.full_like(tau, np.nan), where = np.abs(tau) >= 1e-3)

# Sum of squares error propagation: dv = sqrt[ (ds/t)^2 + (s/t^2 dt)^2 ]
vpol_err_1 = np.divide(
    poloidal_separation_err,
    tau,
    out = np.full_like(tau, np.nan),
    where = np.abs(tau) >= 1e-3)

vpol_err_2 = np.divide(
    poloidal_separation * tau_err,
    tau**2,
    out = np.full_like(tau, np.nan),
    where = np.abs(tau) >= 1e-3
)

vpol_err = np.sqrt(vpol_err_1**2 + vpol_err_2**2)

try:
    os.mkdir(os.path.join(output_path, exp_id))
except FileExistsError:
    print('Folder already exists.')

with h5py.File(os.path.join(output_path, f'{exp_id}', output_file), 'w') as f:
    
    data = f.create_group('ccf_data')
    
    data.create_dataset('time_range', data = time_range)
    data.create_dataset('channels', data = channels)
    data.create_dataset('ccf_max_time_lags', data = tau)
    data.create_dataset('ccf_max_time_lag_err', data = tau_err)
    data.create_dataset('ccf_correlations', data = corrs)
    data.create_dataset('poloidal_velocity', data = vpol)
    data.create_dataset('poloidal_velocity_err', data = vpol_err)
        
    data.attrs['exp_id'] = exp_id
    data.attrs['poloidal_deflection_voltage'] = Udefl
    data.attrs['calibration_factor'] = CALIBRATION_FACTOR
    data.attrs['calibration_factor_err'] = CALIBRATION_FACTOR_ERR
    data.attrs['poloidal_separation'] = poloidal_separation
    data.attrs['bandpass_range'] = bandpass_range
    data.attrs['ccf_window'] = xcorr_window
    data.attrs['ccf_interval_n'] = xcorr_interval
    data.attrs['ccf_fitting_method'] = xcorr_fitting_method
