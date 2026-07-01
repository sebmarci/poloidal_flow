"""
Cross-correlation analysis module for ABES poloidal flow measurements.
"""

import os
import flap
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import correlate
from scipy.optimize import minimize, curve_fit
from typing import List

import matplotlib.pyplot as plt

from .config import CorrelationConfig

def gaussian_func(x, x0, sigma, a, b):
    return a * np.exp(-(x-x0)**2 / (2*sigma**2)) + b

def parabolic_func(x, x0, a, b):
    return a * (x - x0)**2 + b

# General 2 dimensional parabola parameterized with maxima
def parabolic_func_2d(coords, t0, ch0, a, b, c, d):
    t, ch = coords
    return a * (t-t0)**2 + b * (ch-ch0)**2 + c * (t-t0)*(ch-ch0) + d

class CorrelationAnalysis:
    """
    Cross-correlation analysis for ABES poloidal flow measurements.

    This class computes cross-correlation functions between signals from
    different deflection states, extracts time delays using either Gaussian
    fitting or cubic spline interpolation, and provides visualization methods.

    Parameters
    ----------
    data_defl0 : flap.DataObject
        ABES signal data for deflection state 0.
    data_defl1 : flap.DataObject
        ABES signal data for deflection state 1.
    config : CorrelationConfig
        Configuration object containing cross-correlation parameters.

    Attributes
    ----------
    data_defl0 : flap.DataObject
        Signal data for deflection state 0.
    data_defl1 : flap.DataObject
        Signal data for deflection state 1.
    config : CorrelationConfig
        Configuration parameters.
    fitting_methods : dict
        Dictionary mapping method names to fitting functions.
    fitting_method : callable
        Selected fitting method based on configuration.

    Raises
    ------
    ValueError
        If xcorr_fitting_method is not 'gaussian' or 'cubic_spline'.
    """
    
    def __init__(self, data_defl0: flap.DataObject, data_defl1: flap.DataObject, config: CorrelationConfig):
        
        self.exp_id = data_defl0.exp_id
        self.data_defl0 = data_defl0
        self.data_defl1 = data_defl1
        self.config = config
        
        self.fitting_methods = {
            'gaussian': self.fit_gaussian,
            'cubic_spline': self.fit_cubic_spline,
            'parabola': self.fit_parabola
        }
        
        try:
            self.fitting_method = self.fitting_methods[self.config.xcorr_fitting_method]
        except KeyError:
            raise ValueError('Invalid fitting method: must be gaussian or cubic_spline')
        
    def get_poloidal_deflection_voltage(self):
        
        xmlpath = os.path.join(self.config.apdcam_path, self.exp_id, f'{self.exp_id}_config.xml')
                
        xml = flap.FlapXml()
        xml.read_file(xmlpath)
        
        voltage_top = float(xml.get_element(
            section = 'Chopper', 
            element = 'VoltTop1'
        )['Value'])

        voltage_bottom = float(xml.get_element(
            section = 'Chopper',
            element = 'VoltBottom1'
        )['Value'])
        
        return voltage_top - voltage_bottom
        
    def ccf_window_single(self, data0, data1):
        """
        Compute cross-correlation function for a single time window.

        Parameters
        ----------
        data0 : flap.DataObject
            First data object (typically deflection state 0).
        data1 : flap.DataObject
            Second data object (typically deflection state 1).

        Returns
        -------
        ccf_sliced : flap.DataObject
            Cross-correlation function sliced to the relevant time lag range
            (-100 to 120 microseconds by default).

        Notes
        -----
        Uses FLAP's ccf method to compute normalized cross-correlation.
        """
        
        # CCF convention: upwards beam is defl0, downwards is defl1
        # Positive time lag of CCF(lower, upper) means upwards flow
        
        ccf = data1.ccf(
            data0, 
            coordinate = 'Time',
            options = {
                'Interval_n': self.config.xcorr_interval,
                'Resolution': self.config.xcorr_resolution,
                'Normalize': self.config.xcorr_normalize,
                'Range': list(self.config.xcorr_time_lag_interval),
                'Trend removal': None
            }
        )
        
        ccf.get_coordinate_object('Time lag').start += data1.coordinate('Time')[0][0] - data0.coordinate('Time')[0][0]
        
        return ccf
    
    def ccf_2d_window_single(self, defl0, defl1):
        
        time_step = defl0.get_coordinate_object('Time').step[0]
        channel_step = 1

        time_dim_0 = defl0.get_coordinate_object('Time').dimension_list[0]
        time_dim_1 = defl1.get_coordinate_object('Time').dimension_list[0]
        n = min(defl0.data.shape[time_dim_0], defl1.data.shape[time_dim_1])

        sl0 = [slice(None)] * defl0.data.ndim; sl0[time_dim_0] = slice(0, n)
        sl1 = [slice(None)] * defl1.data.ndim; sl1[time_dim_1] = slice(0, n)
        
        data0 = defl0.data[tuple(sl0)]
        data1 = defl1.data[tuple(sl1)]

        data0_norm = (data0 - data0.mean()) / data0.std()
        data1_norm = (data1 - data1.mean()) / data1.std()

        ccf2d = correlate(data1_norm, data0_norm, mode='full') / data0.size

        n_channel, n_time = data0_norm.shape
        channel_lag = np.arange(-(n_channel - 1), n_channel) * channel_step
        time_lag    = np.arange(-(n_time - 1), n_time) * time_step
        
        time_lag += defl1.coordinate('Time')[0].min() - defl0.coordinate('Time')[0].min()
        time_mask = (time_lag >= self.config.xcorr_time_lag_interval[0]) & (time_lag <= self.config.xcorr_time_lag_interval[1])

        # Convert to microseconds in time_lag
        return ccf2d[:, time_mask], channel_lag, time_lag[time_mask]*1e6
    
    def fit_2d_parabola(self, time_lag, ch_lag, ccf_data):
        
        t_idx_window = 4
        ch_idx_window = 2
        
        # ccf_data is laid out as (channel_lag, time_lag)
        ch_max_idx, t_max_idx = np.unravel_index(np.argmax(ccf_data), ccf_data.shape)
        t_idx_range = np.arange(t_max_idx - t_idx_window, t_max_idx + t_idx_window + 1)
        ch_idx_range = np.arange(ch_max_idx - ch_idx_window, ch_max_idx + ch_idx_window + 1)
        idx_sel = np.ix_(ch_idx_range, t_idx_range)

        CH, T = np.meshgrid(ch_lag[ch_idx_range], time_lag[t_idx_range], indexing='ij')

        popt, pcov = curve_fit(
            parabolic_func_2d,
            (T.ravel(), CH.ravel()),
            ccf_data[idx_sel].ravel(),
            p0 = [time_lag[t_max_idx], ch_lag[ch_max_idx], -1, -1, 0, ccf_data.max()],
            )

        perr = np.sqrt(np.diag(pcov))

        return popt[0], popt[1], perr[0], perr[1]
            
    def get_max_lags_2d(self, times, ch_window):
        
        ch_range = np.arange(ch_window + 1, 40 - ch_window + 1)
        
        taumax_arr = np.zeros((len(times), len(ch_range)))
        chmax_arr = np.zeros_like(taumax_arr)
        tauerr_arr = np.zeros_like(taumax_arr)
        cherr_arr = np.zeros_like(taumax_arr)
        
        for (i, t) in enumerate(times):
            
            defl0_time_slice = self.data_defl0.slice_data(
                slicing = {'Time': flap.Intervals(t - self.config.xcorr_window/2, t + self.config.xcorr_window/2)},
            )
            defl1_time_slice = self.data_defl1.slice_data(
                slicing = {'Time': flap.Intervals(t - self.config.xcorr_window/2, t + self.config.xcorr_window/2)},
            )
            
            for (j, ch) in enumerate(ch_range):
                
                print(f'2D CCF: t = {t}, ch = {ch}')
                
                defl0_ch_slice = defl0_time_slice.slice_data(
                    slicing = {'Channel number': flap.Intervals(ch - ch_window, ch + ch_window)}
                )
                
                defl1_ch_slice = defl1_time_slice.slice_data(
                    slicing = {'Channel number': flap.Intervals(ch - ch_window, ch + ch_window)}
                )
                
                ccf2d, ch_lag, time_lag = self.ccf_2d_window_single(defl0_ch_slice, defl1_ch_slice)
                taumax, chmax, tauerr, cherr = self.fit_2d_parabola(time_lag, ch_lag, ccf2d)
                
                taumax_arr[i, j] = taumax
                chmax_arr[i, j] = chmax
                tauerr_arr[i, j] = tauerr
                cherr_arr[i, j] = cherr
                
        return ch_range, taumax_arr, chmax_arr, tauerr_arr, cherr_arr
                
    
    def fit_gaussian(self, ccf):
        """
        Fit Gaussian curve to cross-correlation function.

        Parameters
        ----------
        ccf : flap.DataObject
            Cross-correlation function data.

        Returns
        -------
        tau : float
            Time delay in microseconds (peak position).
        ccf_max : float
            Maximum correlation value at the peak.
        popt : numpy.ndarray
            Optimal fit parameters [x0, sigma, a, b].

        Notes
        -----
        Uses scipy.optimize.curve_fit with initial guess [0, 30, 1, 0]
        for [x0, sigma, a, b] parameters.
        """
        
        p0 = [0, 30, 1, 0]
                
        time_lags = ccf.coordinate('Time lag')[0] * 1e6    
        popt, _ = curve_fit(gaussian_func, time_lags, ccf.data, p0 = p0)
        
        return popt[0], gaussian_func(popt[0], *popt), popt
    
    def fit_cubic_spline(self, ccf):
        """
        Fit cubic spline to cross-correlation function and find peak.

        Parameters
        ----------
        ccf : flap.DataObject
            Cross-correlation function data.

        Returns
        -------
        tau : float
            Time delay in microseconds (peak position).
        ccf_max : float
            Maximum correlation value at the peak.
        cubic_spline : CubicSpline
            Fitted cubic spline object.

        Notes
        -----
        Uses scipy.interpolate.CubicSpline followed by scipy.optimize.minimize
        to find the maximum of the interpolated function.
        """
                        
        time_lags = ccf.coordinate('Time lag')[0] * 1e6  
              
        cubic_spline = CubicSpline(time_lags, ccf.data)        
        optimization = minimize(lambda x: -cubic_spline(x), x0 = 0)
        
        tau = optimization['x'][0]
                
        return tau, cubic_spline(tau), cubic_spline
    
    def fit_parabola(self, ccf):
        
        time_lags = ccf.coordinate('Time lag')[0] * 1e6 
        ind_max = np.argmax(np.abs(ccf.data))
        
        time_lag_slice = time_lags[ind_max-2:ind_max+3]
        ccf_slice = ccf.data[ind_max-2:ind_max+3]
        ccf_err_slice = ccf.error[ind_max-2:ind_max+3]
        
        # popt = [x0, a, b]
        popt, pcov = curve_fit(parabolic_func, time_lag_slice, ccf_slice, sigma = ccf_err_slice)
        perr = np.sqrt(np.diag(pcov))
        
        return popt[0], perr[0], parabolic_func(popt[0], *popt), popt
    
    def get_max_time_lag(self, times, channels):
        """
        Compute time delays for multiple time windows and channels.

        Parameters
        ----------
        times : array_like
            Array of time points (in seconds) at which to compute correlations.
        channels : array_like
            Array of channel numbers to analyze.

        Returns
        -------
        tau_vals : numpy.ndarray
            2D array of time delays in microseconds with shape (len(times), len(channels)).
        corr_vals : numpy.ndarray
            2D array of correlation values at the peaks with shape (len(times), len(channels)).

        Notes
        -----
        For each time point and channel:
        1. Slices data from both deflection states to the configured time window
        2. Computes cross-correlation function
        3. Extracts time delay using the configured fitting method

        Progress is printed to stdout as "t = {time} s, ch = {channel}" for
        each computation.
        """
                                
        tau_vals = np.zeros((len(times), len(channels)))
        tau_err_vals = np.zeros_like(tau_vals)
        corr_vals = np.zeros_like(tau_vals)
        corr_err_vals = np.zeros_like(tau_vals)
        
        for (i, t) in enumerate(times):
            
            defl0_time_slice = self.data_defl0.slice_data(
                slicing = {'Time': flap.Intervals(t - self.config.xcorr_window/2, t + self.config.xcorr_window/2)},
            )
            defl1_time_slice = self.data_defl1.slice_data(
                slicing = {'Time': flap.Intervals(t - self.config.xcorr_window/2, t + self.config.xcorr_window/2)},
            )
            
            # Interleaved chopper grids, closes value slicing can leave defl0
            # and defl1 differing by 1 sample. Truncate to the common length.
            time_dim_0 = defl0_time_slice.get_coordinate_object('Time').dimension_list[0]
            time_dim_1 = defl1_time_slice.get_coordinate_object('Time').dimension_list[0]
            n = min(defl0_time_slice.data.shape[time_dim_0],
                    defl1_time_slice.data.shape[time_dim_1])

            sl0 = [slice(None)] * defl0_time_slice.data.ndim
            sl0[time_dim_0] = slice(0, n)
            defl0_time_slice.data = defl0_time_slice.data[tuple(sl0)]
            defl0_time_slice.shape = defl0_time_slice.data.shape

            sl1 = [slice(None)] * defl1_time_slice.data.ndim
            sl1[time_dim_1] = slice(0, n)
            defl1_time_slice.data = defl1_time_slice.data[tuple(sl1)]
            defl1_time_slice.shape = defl1_time_slice.data.shape
            
            for (j, ch) in enumerate(channels):
                
                print(f't = {t} s, ch = {ch}')
                                
                defl0_single = defl0_time_slice.slice_data(
                    slicing = {'Channel number': ch}
                )
                defl1_single = defl1_time_slice.slice_data(
                    slicing = {'Channel number': ch}
                )
                
                ccf_single = self.ccf_window_single(defl0_single, defl1_single)                
                tau, tau_err, corr, _ = self.fitting_method(ccf_single)
                
                peak_idx = np.argmax(np.abs(ccf_single.data))
                
                tau_vals[i, j] = tau
                tau_err_vals[i, j] = tau_err
                corr_vals[i, j] = corr
                corr_err_vals[i, j] = ccf_single.error[peak_idx] / ccf_single.data[peak_idx]
                            
        return tau_vals, tau_err_vals, corr_vals, corr_err_vals
            
            