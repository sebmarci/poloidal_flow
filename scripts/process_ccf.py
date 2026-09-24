# Calculate CCF time lags and poloidal velocities for one experiment

import os

from poloidal_flow import ABESConfig, ABESDataReader, CorrelationAnalysis, CorrelationConfig
import flap
import flap_w7x_abes
import numpy as np
import h5py
from uncertainties import unumpy as unp

exp_id = '20250409.046'
model = 'elliptical'
xcorr_fitting_method = 'parabola'
xcorr_window = 1 # 1 s
xcorr_interval = 150
bandpass_range = [500, 500]
time_range = [0, 10]

output_path = 'processed_data/elliptical'
output_file = f'{exp_id}_elliptical_highpass_2000.h5'

xconfig = CorrelationConfig(
    xcorr_fitting_method = xcorr_fitting_method,
    xcorr_window = xcorr_window,
    xcorr_interval = xcorr_interval
)

print('Reading')

#defl0, defl1 = ABESDataReader(config).read_data()

defl0 = flap.load('pickled_shot_data/20250409.046_highpass_2000_defl0.pkl')
defl1 = flap.load('pickled_shot_data/20250409.046_highpass_2000_defl1.pkl')

print('Calculating correlations')

anal = CorrelationAnalysis(defl0, defl1, xconfig)
anal.get_deflection_params()

time_range = np.linspace(0, 10, 101)
channels = np.arange(1, 41)

if model == 'taylor':

    tau, corrs = anal.get_max_time_lag_taylor(time_range, channels)
    vpol = anal.get_velocity_taylor(tau)

    try:
        os.mkdir(os.path.join(output_path, exp_id))
    except FileExistsError:
        print('Folder already exists.')

    with h5py.File(os.path.join(output_path, f'{exp_id}', output_file), 'w') as f:

        data = f.create_group('ccf_data')

        data.create_dataset('time_range', data = time_range)
        data.create_dataset('channels', data = channels)
        data.create_dataset('ccf_max_time_lags', data = unp.nominal_values(tau))
        data.create_dataset('ccf_max_time_lag_err', data = unp.std_devs(tau))
        data.create_dataset('ccf_correlations', data = unp.nominal_values(corrs))
        data.create_dataset('ccf_correlations_err', data = unp.std_devs(corrs))
        data.create_dataset('poloidal_velocity', data = unp.nominal_values(vpol))
        data.create_dataset('poloidal_velocity_err', data = unp.std_devs(vpol))

        data.attrs['exp_id'] = exp_id
        data.attrs['model'] = model
        data.attrs['poloidal_deflection_voltage'] = anal.deflection_voltage
        data.attrs['calibration_factor'] = anal.calibration_factor.nominal_value
        data.attrs['calibration_factor_err'] = anal.calibration_factor.std_dev
        data.attrs['poloidal_separation'] = anal.poloidal_deflection.nominal_value
        data.attrs['bandpass_range'] = bandpass_range
        data.attrs['ccf_window'] = xcorr_window
        data.attrs['ccf_interval_n'] = xcorr_interval
        data.attrs['ccf_fitting_method'] = xcorr_fitting_method

elif model == 'elliptical':

    tau, tau0, corrs = anal.get_max_time_lag_elliptical(time_range, channels)
    vpol, vfade = anal.get_velocity_elliptical(tau, tau0)

    try:
        os.mkdir(os.path.join(output_path, exp_id))
    except FileExistsError:
        print('Folder already exists.')

    with h5py.File(os.path.join(output_path, f'{exp_id}', output_file), 'w') as f:

        data = f.create_group('ccf_data')

        data.create_dataset('time_range', data = time_range)
        data.create_dataset('channels', data = channels)
        data.create_dataset('ccf_max_time_lags', data = unp.nominal_values(tau))
        data.create_dataset('ccf_max_time_lag_err', data = unp.std_devs(tau))
        data.create_dataset('tau_0', data = unp.nominal_values(tau0))
        data.create_dataset('tau_0_err', data = unp.std_devs(tau0))
        data.create_dataset('ccf_correlations', data = unp.nominal_values(corrs))
        data.create_dataset('ccf_correlations_err', data = unp.std_devs(corrs))
        data.create_dataset('poloidal_velocity', data = unp.nominal_values(vpol))
        data.create_dataset('poloidal_velocity_err', data = unp.std_devs(vpol))
        data.create_dataset('fading_velocity', data = unp.nominal_values(vfade))
        data.create_dataset('fading_velocity_err', data = unp.std_devs(vfade))


        data.attrs['exp_id'] = exp_id
        data.attrs['model'] = model
        data.attrs['poloidal_deflection_voltage'] = anal.deflection_voltage
        data.attrs['calibration_factor'] = anal.calibration_factor.nominal_value
        data.attrs['calibration_factor_err'] = anal.calibration_factor.std_dev
        data.attrs['poloidal_separation'] = anal.poloidal_deflection.nominal_value
        data.attrs['bandpass_range'] = bandpass_range
        data.attrs['ccf_window'] = xcorr_window
        data.attrs['ccf_interval_n'] = xcorr_interval
        data.attrs['ccf_fitting_method'] = xcorr_fitting_method