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

    Examples
    --------
    >>> config = CorrelationConfig(xcorr_fitting_method='gaussian')
    >>> analyzer = CorrelationAnalysis(data_defl0, data_defl1, config)
    >>> tau_vals, corr_vals = analyzer.get_max_time_lag(times, channels)
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
        chopper frequency being used.
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
            slicing = {'Time lag': flap.Intervals(-1e-4, 1.2e-4)}, # TODO this needs to be adjusted if the chop freq is different !!!
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
        to find the maximum of the interpolated function. Initial guess for
        optimization is 0 microseconds.
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
    
    def plot_ccf_single_gaussian(self, time, channel):
        
        p0 = [0, 30, 1, 0]
    
        data0 = self.data_defl0.slice_data(
            slicing = {'Time': flap.Intervals(time - self.config.xcorr_window/2, time + self.config.xcorr_window/2), 'Channel number': channel},
            options = {'Interpolation': 'Linear'}
        )
        
        data1 = self.data_defl1.slice_data(
            slicing = {'Time': flap.Intervals(time - self.config.xcorr_window/2, time + self.config.xcorr_window/2), 'Channel number': channel},
            options = {'Interpolation': 'Linear'}
        )
        
        ccf = self.ccf_window_single(data0, data1)
                        
        time_lags = ccf.coordinate('Time lag')[0] * 1e6  
        
        popt, pcov = curve_fit(gaussian_func, time_lags, ccf.data, p0 = p0)   
        perr = np.sqrt(np.diag(pcov))
        xopt = popt[0]
        xopt_err = perr[0]
        
        ts = np.linspace(min(time_lags), max(time_lags), 1000)    
        
        plt.figure()
        plt.title(f'ABES single CCF, Gaussian fit, ch = {channel}, t = {time}')
        plt.xlabel('$\\tau$ [$\\mu$s]')
        plt.ylabel('Normalized CCF')
        plt.scatter(time_lags, ccf.data)
        plt.plot(ts, gaussian_func(ts, *popt))
        plt.scatter(xopt, gaussian_func(xopt, *popt))
        
    def plot_ccf_single_spline(self, time, channel):
            
        data0 = self.data_defl0.slice_data(
            slicing = {'Time': flap.Intervals(time - self.config.xcorr_window/2, time + self.config.xcorr_window/2), 'Channel number': channel},
            options = {'Interpolation': 'Linear'}
        )
        
        data1 = self.data_defl1.slice_data(
            slicing = {'Time': flap.Intervals(time - self.config.xcorr_window/2, time + self.config.xcorr_window/2), 'Channel number': channel},
            options = {'Interpolation': 'Linear'}
        )
        
        ccf = self.ccf_window_single(data0, data1)
        
        time_lags = ccf.coordinate('Time lag')[0] * 1e6  
        
        cubic_spline = CubicSpline(time_lags, ccf.data)        
        optimization = minimize(lambda x: -cubic_spline(x), x0 = 0)    
        tau_max = optimization['x']
        
        ts = np.linspace(min(time_lags), max(time_lags), 1000)    
        
        plt.figure()
        plt.title(f'ABES single CCF, spline fit, ch = {channel}, t = {time} \n $\\tau$ = {tau_max} $\\mu$s')
        plt.xlabel('$\\tau$ [$\\mu$s]')
        plt.ylabel('Normalized CCF')
        plt.scatter(time_lags, ccf.data)
        plt.plot(ts, cubic_spline(ts))
        plt.scatter(tau_max, cubic_spline(tau_max))
        
    def plot_ccf_allch(self, time, figsize, method):
        """
        Plot cross-correlation functions for all channels at a single time point.

        Parameters
        ----------
        time : float
            Time point in seconds for the analysis window center.
        figsize : tuple
            Figure size as (width, height) in inches.
        method : str
            Fitting method to use: 'spline' or 'gaussian'.

        Notes
        -----
        Creates a subplot grid (8 rows x 5 columns) showing cross-correlation
        functions for all 40 ABES channels. Each subplot displays:
        - Scatter plot of CCF data points
        - Fitted curve (spline or Gaussian)
        - Peak position marker and vertical line
        - Time delay value in the subplot title

        The main title indicates the fitting method and time point.
        """
        
        fig, axes = plt.subplots(nrows = 8, ncols = 5, figsize = figsize)
        plt.suptitle(f'Cubic spline CCF fits, single time slice, all channels, t = {time} s', fontsize = 18)
        
        axes = axes.flat
        
        data0 = self.data_defl0.slice_data(
            slicing = {'Time': flap.Intervals(time - self.config.xcorr_window/2, time + self.config.xcorr_window/2)},
            options = {'Interpolation': 'Linear'}
        )
        
        data1 = self.data_defl1.slice_data(
            slicing = {'Time': flap.Intervals(time - self.config.xcorr_window/2, time + self.config.xcorr_window/2)},
            options = {'Interpolation': 'Linear'}
        )
        
        for (i, ch) in enumerate(range(1, 41)):
            
            data0_ch = data0.slice_data(
                slicing = {'Channel number': ch}
            )
            
            data1_ch = data1.slice_data(
                slicing = {'Channel number': ch}
            )
                        
            ccf = self.ccf_window_single(data0_ch, data1_ch)
            time_lags = ccf.coordinate('Time lag')[0] * 1e6 
            ts = np.linspace(min(time_lags), max(time_lags), 1000)     
            
            if method == 'spline':
                taumax, ccfmax, cs = self.fit_cubic_spline(ccf)
                axes[i].plot(ts, cs(ts))
            elif method == 'gaussian':
                taumax, ccfmax, popt = self.fit_gaussian(ccf)
                axes[i].plot(ts, gaussian_func(ts, *popt))
                       
            axes[i].set_title(f'CH{i}, $\\Delta \\tau$ = {taumax} $\\mu$s')
            axes[i].set_xlabel('$\\tau$ [$\\mu$s]')
            axes[i].set_ylabel('Cross-correlation')       
            axes[i].scatter(time_lags, ccf.data)
            axes[i].scatter(taumax, ccfmax)
            axes[i].axvline(x = taumax)     
            
            