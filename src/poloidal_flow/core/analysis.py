"""
Cross-correlation analysis module for ABES poloidal flow measurements.
"""

import flap
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, curve_fit
from typing import List

import matplotlib.pyplot as plt

from .config import CorrelationConfig

def gaussian_func(x, x0, sigma, a, b):
    return a * np.exp((x-x0)**2 / (2*sigma**2)) + b

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
        
        self.data_defl0 = data_defl0
        self.data_defl1 = data_defl1
        self.config = config
        
        self.fitting_methods = {
            'gaussian': self.fit_gaussian,
            'cubic_spline': self.fit_cubic_spline
        }
        
        try:
            self.fitting_method = self.fitting_methods[self.config.xcorr_fitting_method]
        except KeyError:
            raise ValueError('Invalid fitting method: must be gaussian or cubic_spline')
        
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
        The time lag range is currently hardcoded but should match the
        beam chop frequency being used.
        """
        
        ccf = data0.ccf(
            data1, 
            coordinate = 'Time',
            options = {
                'Interval_n': self.config.xcorr_interval,
                'Resolution': self.config.xcorr_resolution,
                'Normalize': self.config.xcorr_normalize
            }
        )
        
        ccf_sliced = ccf.slice_data(
            slicing = {'Time lag': flap.Intervals(*self.config.xcorr_time_lag_interval)},
        )
        
        return ccf_sliced
    
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
        corr_vals = np.zeros_like(tau_vals)
        
        for (i, t) in enumerate(times):
            
            defl0_time_slice = self.data_defl0.slice_data(
                slicing = {'Time': flap.Intervals(t - self.config.xcorr_window/2, t + self.config.xcorr_window/2)},
                options = {'Interpolation': 'Linear'}
            )
            defl1_time_slice = self.data_defl1.slice_data(
                slicing = {'Time': flap.Intervals(t - self.config.xcorr_window/2, t + self.config.xcorr_window/2)},
                options = {'Interpolation': 'Linear'}
            )
            
            for (j, ch) in enumerate(channels):
                
                print(f't = {t} s, ch = {ch}')
                                
                defl0_single = defl0_time_slice.slice_data(
                    slicing = {'Channel number': ch}
                )
                defl1_single = defl1_time_slice.slice_data(
                    slicing = {'Channel number': ch}
                )
                
                ccf_single = self.ccf_window_single(defl0_single, defl1_single)
                tau, corr, _ = self.fitting_method(ccf_single)
                
                tau_vals[i, j] = tau
                corr_vals[i, j] = corr
                            
        return tau_vals, corr_vals
            
            