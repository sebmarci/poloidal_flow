"""
Cross-correlation analysis module for ABES poloidal flow measurements.

The two deflection states of the ABES beam are separated poloidally by a
distance ``s``, so a poloidal flow shifts their signals in time relative to
one another. This module extracts that shift from the cross-correlation
function (CCF) of the two signals and converts it into a velocity with either
of the two turbulence models of Krämer-Flecken et al. 2025 (Plasma Phys.
Control. Fusion 67 055024):

- Taylor, frozen turbulence, equation (1): ``v_pol = s / dt``
- Elliptical approach (EA), equations (2) and (3), which additionally account
  for the decay of the turbulence through a fading timescale ``tau_0``:
  ``v_pol = s * dt / (dt^2 + tau_0^2)`` and ``v_fade = s / sqrt(dt^2 + tau_0^2)``

Here ``dt`` is the lag of the CCF maximum and ``tau_0`` the lag at which the
mean autocorrelation function (ACF) of the two states equals the CCF maximum.
Taylor is the ``tau_0 -> 0`` limit of the EA and is adequate only while
``dt / tau_0 >= 5``; W7-X generally sits below that, so the EA is the model of
interest.

Time lags are handled in seconds inside FLAP and returned in microseconds;
combined with the beam separation in mm this makes all velocities km/s.
"""

import os
import flap
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import correlate
from scipy.optimize import minimize, curve_fit, fsolve
from typing import List

import matplotlib.pyplot as plt

from .config import CorrelationConfig

CALIBRATION_FACTOR = 0.09477279 # mm/V
CALIBRATION_FACTOR_ERR = 0.01016466 # mm/V

def gaussian_func(x, x0, sigma, a, b):
    """Gaussian with peak position `x0`, width `sigma`, amplitude `a`, offset `b`."""
    return a * np.exp(-(x-x0)**2 / (2*sigma**2)) + b

def parabolic_func(x, x0, a, b):
    """Parabola parameterized by its extremum position `x0` and value `b`."""
    return a * (x - x0)**2 + b

# General 2 dimensional parabola parameterized with maxima
def parabolic_func_2d(coords, t0, ch0, a, b, c, d):
    """
    2D parabola parameterized by its extremum, for the deprecated 2D CCF fit.

    `coords` is a ``(t, ch)`` tuple of time lag and channel lag arrays, `t0` and
    `ch0` locate the extremum, `c` is the cross term and `d` the extremum value.
    """
    t, ch = coords
    return a * (t-t0)**2 + b * (ch-ch0)**2 + c * (t-t0)*(ch-ch0) + d

class CorrelationAnalysis:
    """
    Cross-correlation analysis for ABES poloidal flow measurements.

    Computes CCFs and ACFs between the two deflection states over a grid of
    time windows and channels, extracts the CCF maximum time lag by fitting the
    peak, and converts the lags into poloidal velocities with the Taylor or the
    elliptical model.

    Parameters
    ----------
    data_defl0 : flap.DataObject
        ABES signal data for deflection state 0 (upwards deflected beam).
    data_defl1 : flap.DataObject
        ABES signal data for deflection state 1 (downwards deflected beam).
    config : CorrelationConfig
        Configuration object containing cross-correlation parameters.

    Attributes
    ----------
    exp_id : str
        Experiment ID, taken from `data_defl0`.
    data_defl0, data_defl1 : flap.DataObject
        Signal data for the two deflection states.
    config : CorrelationConfig
        Configuration parameters.
    calibration_factor, calibration_factor_err : float
        Copies of the module-level beam deflection calibration, mm/V.
    fitting_methods : dict
        Maps method names to the corresponding peak fitting methods.
    fitting_method : callable
        The peak fitting method selected by ``config.xcorr_fitting_method``.
    deflection_voltage : float
        Chopper deflection voltage in V. Set by ``get_deflection_params``.
    poloidal_deflection, poloidal_deflection_err : float
        Poloidal beam separation `s` and its uncertainty in mm. Set by
        ``get_deflection_params``.

    Raises
    ------
    ValueError
        If ``config.xcorr_fitting_method`` is not one of 'gaussian',
        'cubic_spline' or 'parabola'.

    Notes
    -----
    Only 'parabola' can be used with the ``get_max_time_lag_*`` methods; the
    other two are deprecated and survive for `CCFPlotter` only.

    Examples
    --------
    >>> analyzer = CorrelationAnalysis(data_defl0, data_defl1, corr_config)
    >>> tau, tau_err, tau0, corr = analyzer.get_max_time_lag_elliptical(
    ...     np.linspace(0, 10, 101), np.arange(1, 41))
    >>> vpol, vpol_err, vfade, vfade_err = analyzer.get_velocity_elliptical(
    ...     tau, tau_err, tau0)
    """

    def __init__(self, data_defl0: flap.DataObject, data_defl1: flap.DataObject, config: CorrelationConfig):
        
        self.exp_id = data_defl0.exp_id
        self.data_defl0 = data_defl0
        self.data_defl1 = data_defl1
        self.config = config
        self.calibration_factor = CALIBRATION_FACTOR
        self.calibration_factor_err = CALIBRATION_FACTOR_ERR
        
        self.fitting_methods = {
            'gaussian': self.fit_gaussian,
            'cubic_spline': self.fit_cubic_spline,
            'parabola': self.fit_parabola
        }
        
        try:
            self.fitting_method = self.fitting_methods[self.config.xcorr_fitting_method]
        except KeyError:
            raise ValueError('Invalid fitting method: must be gaussian, cubic_spline or parabola')

    def get_deflection_params(self):
        """
        Read the beam deflection from the APDCAM XML configuration of the shot.

        Sets ``deflection_voltage`` (the top minus bottom chopper plate voltage,
        V) and turns it into the poloidal beam separation `s` with the
        module-level calibration, storing ``poloidal_deflection`` and
        ``poloidal_deflection_err`` in mm.

        Notes
        -----
        The XML is looked up as ``{config.apdcam_path}/{exp_id}/{exp_id}_config.xml``.
        Called lazily by the ``get_velocity_*`` methods, so it rarely needs to be
        called by hand.
        """

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
        
        self.deflection_voltage = voltage_top - voltage_bottom
        self.poloidal_deflection = CALIBRATION_FACTOR * self.deflection_voltage
        self.poloidal_deflection_err = CALIBRATION_FACTOR_ERR * self.deflection_voltage
    
    def truncate_data(self, data0, data1):
        """
        Trim two data objects in place to a common number of time samples.

        Parameters
        ----------
        data0, data1 : flap.DataObject
            Time-sliced deflection states. Both are modified in place: the
            longer one is cut from the end to the length of the shorter one.

        Notes
        -----
        FLAP generates coordinate arrays from ``DataObject.shape`` rather than
        from ``DataObject.data.shape``, so both are updated here. Leaving a
        stale ``.shape`` makes ``coordinate('Time')`` return more samples than
        the data holds.
        """

        # Interleaved chopper grids, closes value slicing can leave defl0
        # and defl1 differing by 1 sample. Truncate to the common length.
        time_dim_0 = data0.get_coordinate_object('Time').dimension_list[0]
        time_dim_1 = data1.get_coordinate_object('Time').dimension_list[0]
        n = min(data0.data.shape[time_dim_0],
                data1.data.shape[time_dim_1])

        sl0 = [slice(None)] * data0.data.ndim
        sl0[time_dim_0] = slice(0, n)
        data0.data = data0.data[tuple(sl0)]
        data0.shape = data0.data.shape

        sl1 = [slice(None)] * data1.data.ndim
        sl1[time_dim_1] = slice(0, n)
        data1.data = data1.data[tuple(sl1)]
        data1.shape = data1.data.shape
        
    def ccf_window_single(self, data0, data1):
        """
        Compute the cross-correlation function for a single time window.

        Parameters
        ----------
        data0 : flap.DataObject
            Reference signal: the CCF is computed as ``data1.ccf(data0)``.
        data1 : flap.DataObject
            Processed signal.

        Returns
        -------
        ccf : flap.DataObject
            Cross-correlation function over the lag range configured by
            ``xcorr_time_lag_interval`` (-100 to 120 us by default), with a
            'Time lag' coordinate in seconds and per-lag errors in ``.error``
            from the scatter over the subintervals.

        Notes
        -----
        The argument order fixes the sign of the time lag and therefore of the
        velocity. The ``get_max_time_lag_*`` methods call this as
        ``ccf_window_single(defl1, defl0)``, so the CCF is computed as
        ``defl0.ccf(defl1)``; swapping the arguments flips the sign of every
        resulting velocity.

        The 'Time lag' axis is shifted by the difference of the two window start
        times, which corrects for the interleaved chopper grids of the two
        deflection states offsetting the windows by a fraction of a sample.
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
                'Trend removal': None,
                'Verbose': False
            }
        )
        
        ccf.get_coordinate_object('Time lag').start += data1.coordinate('Time')[0][0] - data0.coordinate('Time')[0][0]
        
        return ccf
    
    def acf_window_single(self, data):
        """
        Compute the autocorrelation function for a single time window.

        Parameters
        ----------
        data : flap.DataObject
            Single-channel, time-sliced signal.

        Returns
        -------
        acf : flap.DataObject
            Autocorrelation function over the same lag range as
            ``ccf_window_single``, normalized to 1 at zero lag.

        Notes
        -----
        Used by ``get_max_time_lag_elliptical``: the mean ACF of the two
        deflection states defines the elliptical timescale ``tau_0``.

        FLAP's ``ccf`` is called without a reference object. Passing `data` as
        its own reference takes a branch that computes two extra FFT
        autocorrelations, giving identical results (to 2e-16) at roughly twice
        the runtime.
        """
        
        # No reference: FLAP correlates the data object with itself, which is
        # the same ACF as passing data as the reference but roughly 2x faster.
        acf = data.ccf(
            coordinate = 'Time',
            options = {
                'Interval_n': self.config.xcorr_interval,
                'Resolution': self.config.xcorr_resolution,
                'Normalize': self.config.xcorr_normalize,
                'Range': list(self.config.xcorr_time_lag_interval),
                'Trend removal': None,
                'Verbose': False
            }
        )
        
        return acf
    
    def fit_gaussian(self, ccf):
        """
        Fit a Gaussian to the whole cross-correlation function. **Deprecated.**

        Parameters
        ----------
        ccf : flap.DataObject
            Cross-correlation function data.

        Returns
        -------
        tau : float
            Time delay in microseconds (peak position).
        ccf_max : float
            Correlation value at the peak.
        popt : numpy.ndarray
            Optimal fit parameters [x0, sigma, a, b].

        Notes
        -----
        Deprecated: this 3-tuple is incompatible with the 4-value unpacking in
        the ``get_max_time_lag_*`` methods, so selecting 'gaussian' in
        `CorrelationConfig` breaks them. Kept because `CCFPlotter` can still
        draw it. Use ``fit_parabola`` instead.

        Fits over the full retained lag range with initial guess
        ``[0, 30, 1, 0]`` and gives no uncertainty on the peak position.
        """
        
        p0 = [0, 30, 1, 0]
                
        time_lags = ccf.coordinate('Time lag')[0] * 1e6    
        popt, _ = curve_fit(gaussian_func, time_lags, ccf.data, p0 = p0)
        
        return popt[0], gaussian_func(popt[0], *popt), popt
    
    def fit_cubic_spline(self, ccf):
        """
        Interpolate the CCF with a cubic spline and find its peak. **Deprecated.**

        Parameters
        ----------
        ccf : flap.DataObject
            Cross-correlation function data.

        Returns
        -------
        tau : float
            Time delay in microseconds (peak position).
        ccf_max : float
            Correlation value at the peak.
        cubic_spline : scipy.interpolate.CubicSpline
            The fitted spline.

        Notes
        -----
        Deprecated for the same reason as ``fit_gaussian``: a 3-tuple return and
        no peak uncertainty. Kept for `CCFPlotter` only.

        The peak is found by ``scipy.optimize.minimize`` on the negated spline
        starting from zero lag, so it can settle on a local maximum.
        """
                        
        time_lags = ccf.coordinate('Time lag')[0] * 1e6  
              
        cubic_spline = CubicSpline(time_lags, ccf.data)        
        optimization = minimize(lambda x: -cubic_spline(x), x0 = 0)
        
        tau = optimization['x'][0]
                
        return tau, cubic_spline(tau), cubic_spline
    
    def fit_parabola(self, ccf):
        """
        Fit a parabola to the CCF peak. The method to use for time lags.

        Parameters
        ----------
        ccf : flap.DataObject
            Cross-correlation function data, with per-lag errors in ``.error``.

        Returns
        -------
        tau : float
            Peak position (time delay) in microseconds, or nan if no fit was
            possible.
        tau_err : float
            1-sigma uncertainty on `tau` in microseconds, from the fit
            covariance, or nan.
        ccf_max : float
            Fitted parabola value at the peak, or nan.
        popt : numpy.ndarray or None
            Fit parameters ``[x0, a, b]`` of ``parabolic_func``, or None if no
            fit was possible.

        Notes
        -----
        Fits ``a * (x - x0)^2 + b`` to the 5 samples centred on
        ``argmax(|CCF|)``, weighted by the CCF errors. Unlike the deprecated
        methods this yields a propagated uncertainty on the peak position.

        Two cases return all-nan (with `popt` None) instead of raising: a peak
        within 2 samples of the edge of the retained lag interval, which means
        the true maximum is most likely outside it, and a non-converging fit.
        `nan` results propagate into the velocities, but they make
        ``get_max_time_lag_elliptical`` raise at the ``tau_0`` root find.

        The CCF errors are all zero when ``xcorr_interval == 1``, in which case
        ``curve_fit`` silently returns the initial guess instead of a fit.
        """

        time_lags = ccf.coordinate('Time lag')[0] * 1e6
        ind_max = np.argmax(np.abs(ccf.data))

        # The 3-parameter parabola needs 2 samples on both sides of the peak. A
        # peak at the edge of the retained time lag interval means the true
        # maximum is most likely outside it, so no delay can be measured. The
        # slices below would also be empty (negative start index wraps around).
        if ind_max < 2 or ind_max > len(ccf.data) - 3:
            print(f'CCF peak at time lag index {ind_max} of {len(ccf.data)} '
                  '(edge of time lag interval), skipping parabola fit')
            return np.nan, np.nan, np.nan, None

        time_lag_slice = time_lags[ind_max-2:ind_max+3]
        ccf_slice = ccf.data[ind_max-2:ind_max+3]
        ccf_err_slice = ccf.error[ind_max-2:ind_max+3]

        # popt = [x0, a, b]
        try:
            popt, pcov = curve_fit(parabolic_func, time_lag_slice, ccf_slice, sigma = ccf_err_slice)
        except RuntimeError as err:
            print(f'Parabola fit did not converge: {err}')
            return np.nan, np.nan, np.nan, None

        perr = np.sqrt(np.diag(pcov))

        return popt[0], perr[0], parabolic_func(popt[0], *popt), popt
    
    def get_max_time_lag_taylor(self, times, channels):
        """
        CCF maximum time lags for a grid of time windows and channels.

        The quantities needed by the Taylor (frozen turbulence) model.

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
        tau_err_vals : numpy.ndarray
            2D array of 1-sigma uncertainties on the time delays, same shape.
        corr_vals : numpy.ndarray
            2D array of correlation values at the peaks with shape (len(times), len(channels)).

        Notes
        -----
        For each time point and channel:
        1. Slice both deflection states to a window of ``config.xcorr_window``
           centred on the time point, and truncate them to a common length
        2. Compute the CCF of the channel
        3. Extract the peak with the configured fitting method

        The window is sliced in time first and in channel second; pre-slicing
        the channels outside the time loop is slower.

        Feed the results to ``get_velocity_taylor`` for a velocity. Progress is
        printed to stdout as "t = {time} s, ch = {channel}".
        """
                                
        tau_vals = np.zeros((len(times), len(channels)))
        tau_err_vals = np.zeros_like(tau_vals)
        corr_vals = np.zeros_like(tau_vals)

        for (i, t) in enumerate(times):

            defl0_time_slice = self.data_defl0.slice_data(
                slicing = {'Time': flap.Intervals(t - self.config.xcorr_window/2, t + self.config.xcorr_window/2)},
            )
            defl1_time_slice = self.data_defl1.slice_data(
                slicing = {'Time': flap.Intervals(t - self.config.xcorr_window/2, t + self.config.xcorr_window/2)},
            )
            
            self.truncate_data(defl0_time_slice, defl1_time_slice)
                        
            for (j, ch) in enumerate(channels):
                
                print(f't = {t} s, ch = {ch}')
                                
                defl0_single = defl0_time_slice.slice_data(
                    slicing = {'Channel number': ch}
                )
                defl1_single = defl1_time_slice.slice_data(
                    slicing = {'Channel number': ch}
                )
                
                ccf = self.ccf_window_single(defl1_single, defl0_single)
                tau, tau_err, corr, _ = self.fitting_method(ccf)

                tau_vals[i, j] = tau
                tau_err_vals[i, j] = tau_err
                corr_vals[i, j] = corr

        return tau_vals, tau_err_vals, corr_vals
    
    def get_max_time_lag_elliptical(self, times, channels):
        """
        CCF time lags and fading timescales for a grid of windows and channels.

        The quantities needed by the elliptical approach: on top of the CCF
        maximum lag `dt` it returns `tau_0`, the lag at which the mean ACF of
        the two deflection states equals the CCF maximum value.

        Parameters
        ----------
        times : array_like
            Time points in seconds at which to compute the correlations.
        channels : array_like
            Channel numbers to analyze (1-40).

        Returns
        -------
        tau_vals : numpy.ndarray
            CCF maximum time lags in microseconds, shape
            ``(len(times), len(channels))``.
        tau_err_vals : numpy.ndarray
            1-sigma uncertainties on the time lags, same shape.
        tau0_vals : numpy.ndarray
            Fading timescales `tau_0` in microseconds, same shape. Always
            positive: the ACF is symmetric and only ``tau_0^2`` is physical.
        corr_vals : numpy.ndarray
            CCF values at the peaks, same shape.

        Raises
        ------
        RuntimeError
            If the ``tau_0`` root find fails to converge for any point. This
            aborts the whole run rather than skipping one channel; a `nan`
            `corr` from a failed or edge parabola fit triggers it too.

        Notes
        -----
        Per time point and channel:
        1. Slice both deflection states to the window, truncate to a common length
        2. Compute the ACF of each state and average them
        3. Compute the CCF and fit the peak, giving `dt` and `corr`
        4. Cubic-spline the mean ACF and solve ``ACF(tau_0) = corr`` for `tau_0`

        ``fsolve`` returns its initial guess (20 us) on failure, which is
        indistinguishable from a genuine 20 us root, so the solver status is
        checked and turned into the `RuntimeError` above. Even a converged
        solve is not guaranteed to be meaningful: the spline extrapolates
        outside the lag window, so a far spurious root is possible.

        Costs roughly 2-3x ``get_max_time_lag_taylor``, because of the two extra
        correlations per point. Feed the results to ``get_velocity_elliptical``,
        which returns both the convective and the fading velocity.
        """

        tau_vals = np.zeros((len(times), len(channels)))
        tau_err_vals = np.zeros_like(tau_vals)
        corr_vals = np.zeros_like(tau_vals)
        tau0_vals = np.zeros_like(tau_vals)
        
        for (i, t) in enumerate(times):
        
            defl0_time_slice = self.data_defl0.slice_data(
                slicing = {'Time': flap.Intervals(t - self.config.xcorr_window/2, t + self.config.xcorr_window/2)},
            )
            defl1_time_slice = self.data_defl1.slice_data(
                slicing = {'Time': flap.Intervals(t - self.config.xcorr_window/2, t + self.config.xcorr_window/2)},
            )
                        
            self.truncate_data(defl0_time_slice, defl1_time_slice)
            
            for (j, ch) in enumerate(channels):
                
                defl0_single = defl0_time_slice.slice_data(
                    slicing = {'Channel number': ch}
                )
                defl1_single = defl1_time_slice.slice_data(
                    slicing = {'Channel number': ch}
                )
                
                acf0 = self.acf_window_single(defl0_single)
                acf1 = self.acf_window_single(defl1_single)
                acfmean = (acf0.data + acf1.data) / 2
                
                ccf = self.ccf_window_single(defl1_single, defl0_single)
                tau, tau_err, corr, _ = self.fitting_method(ccf)
                
                acf_interp = CubicSpline(acf0.coordinate('Time lag')[0], acfmean)

                # On failure fsolve silently returns x0, which is indistinguishable
                # from a genuine 20 us root, so the solver status has to be checked.
                root, _, ier, mesg = fsolve(
                    lambda x: acf_interp(x) - corr,
                    x0 = 20e-6,
                    full_output = True
                )

                if ier != 1:
                    raise RuntimeError(
                        f'tau0 root finding did not converge at t = {t} s, ch = {ch} '
                        f'(CCF max = {corr}, returned {root[0]*1e6} us): {mesg.strip()}'
                    )

                tau0 = abs(root[0] * 1e6)
                tau_vals[i, j] = tau
                tau_err_vals[i, j] = tau_err
                corr_vals[i, j] = corr
                tau0_vals[i, j] = tau0
                
        return tau_vals, tau_err_vals, tau0_vals, corr_vals
    
    def get_velocity_taylor(self, tau, tau_err):
        """
        Poloidal velocity in the Taylor (frozen turbulence) model.

        Implements equation (1) of Krämer-Flecken et al. 2025, v_pol = s / dt,
        where s is the poloidal beam separation and dt the CCF maximum time lag.

        Parameters
        ----------
        tau : numpy.ndarray
            CCF maximum time lags in microseconds, as returned by
            ``get_max_time_lag_taylor``.
        tau_err : numpy.ndarray
            1-sigma uncertainties on `tau` in microseconds.

        Returns
        -------
        vpol : numpy.ndarray
            Poloidal velocity in km/s (mm/us), same shape as `tau`. Points with
            |tau| < 1e-3 us are set to nan.
        vpol_err : numpy.ndarray
            1-sigma uncertainty on `vpol` in km/s, propagated from the beam
            separation and time lag uncertainties.

        Notes
        -----
        Reads the deflection voltage from the APDCAM config on first use via
        ``get_deflection_params``.
        """

        if not hasattr(self, 'poloidal_deflection'):
            self.get_deflection_params()

        # Infinity blowups can occur here if the CCF max time lag is close to zero.
        # Have to be careful by using np.divide instead of native division

        vpol = np.divide(self.poloidal_deflection, tau, out = np.full_like(tau, np.nan), where = np.abs(tau) >= 1e-3)

        # Sum of squares error propagation: dv = sqrt[ (ds/t)^2 + (s/t^2 dt)^2 ]
        vpol_err_1 = np.divide(
            self.poloidal_deflection_err,
            tau,
            out = np.full_like(tau, np.nan),
            where = np.abs(tau) >= 1e-3)

        vpol_err_2 = np.divide(
            self.poloidal_deflection * tau_err,
            tau**2,
            out = np.full_like(tau, np.nan),
            where = np.abs(tau) >= 1e-3
        )
        
        vpol_err = np.sqrt(vpol_err_1**2 + vpol_err_2**2)
        
        return vpol, vpol_err
    
    def get_velocity_elliptical(self, tau, tau_err, tau0):
        """
        Convective and fading velocity in the elliptical approach.

        Implements equations (2) and (3) of Krämer-Flecken et al. 2025,

            v_pol  = s * dt / (dt^2 + tau_0^2)
            v_fade = s / sqrt(dt^2 + tau_0^2)

        which share the fading time dt_f = sqrt(dt^2 + tau_0^2) of the paper:
        the timescale on which the turbulence decorrelates, combining
        propagation (dt) and decay (tau_0). `v_pol` reduces to the Taylor result
        (equation 1) in the tau_0 -> 0 limit. `v_fade` carries no sign
        information, since dt only enters it squared.

        Parameters
        ----------
        tau : numpy.ndarray
            CCF maximum time lags in microseconds.
        tau_err : numpy.ndarray
            1-sigma uncertainties on `tau` in microseconds. Used for
            `vfade_err` only, see Notes.
        tau0 : numpy.ndarray
            Fading timescales in microseconds, as returned by
            ``get_max_time_lag_elliptical``.

        Returns
        -------
        vpol : numpy.ndarray
            Convective poloidal velocity in km/s (mm/us), same shape as `tau`.
        vpol_err : numpy.ndarray
            Zero placeholder array, see Notes.
        vfade : numpy.ndarray
            Fading velocity in km/s, same shape.
        vfade_err : numpy.ndarray
            1-sigma uncertainty on `vfade` in km/s. Incomplete, see Notes.

        Notes
        -----
        The two velocities are returned together because they are two readings
        of the same fit and differ only in how the fading time enters.

        `tau0` is strictly positive, so the fading time cannot vanish and the
        divisions need no zero guard, unlike the one in ``get_velocity_taylor``.
        A `nan` `tau` still propagates into both velocities.

        Neither error is complete, because `tau0` has no uncertainty estimate.
        For `v_pol` that term is the dominant one, so no propagation is
        attempted at all and `vpol_err` is a zero array rather than a real error
        bar. For `v_fade` the beam separation and `tau` terms are propagated,
        which makes `vfade_err` a lower bound on the true uncertainty.
        """

        if not hasattr(self, 'poloidal_deflection'):
            self.get_deflection_params()

        # tau0 is strictly positive, so the fading time cannot vanish and none
        # of the divisions below need the np.divide guard get_velocity_taylor has.
        fading_time = np.sqrt(tau**2 + tau0**2)

        # Eq. (2): v_pol = s * dt / (dt^2 + tau_0^2)
        vpol = self.poloidal_deflection * tau / fading_time**2

        # TODO IMPLEMENT ERROR PROPAGATION
        vpol_err = np.zeros_like(vpol)

        # Eq. (3): v_fade = s / sqrt(dt^2 + tau_0^2)
        vfade = self.poloidal_deflection / fading_time

        # Sum of squares error propagation:
        # dv = sqrt[ (ds/dtf)^2 + (s dt/dtf^3 ddt)^2 ]. The tau_0 term is absent.
        vfade_err_1 = self.poloidal_deflection_err / fading_time
        vfade_err_2 = self.poloidal_deflection * tau * tau_err / fading_time**3

        vfade_err = np.sqrt(vfade_err_1**2 + vfade_err_2**2)

        return vpol, vpol_err, vfade, vfade_err


    # --- THE FOLLOWING FUNCTIONS ARE UNUSED AND/OR DEPRECATED ---
    # They are only kept for the sake of ease to use when the project demands it (they will probably not be used anyway)
    # Nothing below is part of the poloidal flow analysis; do not extend it.

    def calculate_radial_correlation(self, defl, time, chref):
        """
        CCF of every channel against a reference channel. **Deprecated.**

        Parameters
        ----------
        defl : int
            Deflection state to use, 0 or 1.
        time : float
            Ignored. The method correlates the full time range of the data
            object, not a window around `time`.
        chref : int
            Reference channel number.

        Returns
        -------
        numpy.ndarray
            CCF data for channels 1-40, shape ``(40, n_lags)``.
        """

        data = self.data_defl0 if defl == 0 else self.data_defl1
        ccf_points = []
        
        ref_channel_slice = data.slice_data(
            slicing = {'Channel number': chref}
        )
        
        for ch in range(1, 41):
            
            channel_slice = data.slice_data(
                slicing = {'Channel number': ch}
            )
            
            ccf = self.ccf_window_single(ref_channel_slice, channel_slice)
            ccf_points.append(ccf.data)
            
        return np.array(ccf_points)
    
    def fit_radial_velocity(self, defl, times, chref, chwindow, dev_r):
        """
        Radial velocity from the slope of lag versus major radius. **Deprecated.**

        Parameters
        ----------
        defl : int
            Deflection state to use, 0 or 1.
        times : array_like
            Time points in seconds.
        chref : int
            Reference channel of the radial correlation.
        chwindow : int
            Number of channels on either side of `chref` included in the fit.
        dev_r : array_like
            Device R coordinate of each channel, from
            ``ABESDataReader.read_spatial_calibration``.

        Returns
        -------
        numpy.ndarray
            Radial velocity per time point, as the reciprocal slope of a linear
            fit of CCF maximum lag against major radius.
        """

        time_lags = np.arange(*self.config.xcorr_time_lag_interval, self.data_defl0.get_coordinate_object('Time').step[0])
        radial_velocities = []
        
        for t in times:
            
            print(f't = {t} s')
            
            ccf_data = self.calculate_radial_correlation(defl, t, chref)
            
            maxidx = np.argmax(ccf_data, axis = 1)
            max_t = time_lags[maxidx]

            idx_fit = np.arange(chref - chwindow - 1, chref + chwindow)
            r_fit = dev_r[idx_fit]
            t_fit = max_t[idx_fit]
            p = np.polyfit(r_fit, t_fit, 1)
                        
            radial_velocities.append(1 / p[0])
            
        return np.array(radial_velocities)
    
    def ccf_2d_window_single(self, defl0, defl1):
        """
        2D CCF over time lag and channel lag. **Deprecated.**

        Parameters
        ----------
        defl0, defl1 : flap.DataObject
            Time- and channel-sliced deflection states, laid out as
            ``(channel, time)``.

        Returns
        -------
        ccf2d : numpy.ndarray
            2D CCF, shape ``(channel lag, time lag)``, masked to the configured
            time lag interval.
        channel_lag : numpy.ndarray
            Channel lag axis in channel numbers.
        time_lag : numpy.ndarray
            Time lag axis in microseconds.

        Notes
        -----
        Uses ``scipy.signal.correlate`` rather than FLAP, so it provides no
        error estimate.
        """

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
        """
        Fit a 2D parabola to the peak of a 2D CCF. **Deprecated.**

        Parameters
        ----------
        time_lag, ch_lag : numpy.ndarray
            Time lag (us) and channel lag axes of `ccf_data`.
        ccf_data : numpy.ndarray
            2D CCF laid out as ``(channel lag, time lag)``.

        Returns
        -------
        t0, ch0 : float
            Peak position in time lag (us) and channel lag.
        t0_err, ch0_err : float
            1-sigma uncertainties on the peak position.

        Notes
        -----
        Fits ``parabolic_func_2d`` to a 9x5 (time x channel) index window around
        the maximum. The window is not bounds-checked, so a peak near the edge
        of the lag interval raises or silently wraps around.
        """

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
        """
        2D CCF peak positions over a grid of time windows. **Deprecated.**

        Parameters
        ----------
        times : array_like
            Time points in seconds.
        ch_window : int
            Number of channels on either side of the centre channel included in
            each 2D CCF.

        Returns
        -------
        ch_range : numpy.ndarray
            Centre channels analyzed, i.e. those with a full channel window.
        taumax_arr, chmax_arr : numpy.ndarray
            Peak time lag (us) and channel lag, shape
            ``(len(times), len(ch_range))``.
        tauerr_arr, cherr_arr : numpy.ndarray
            1-sigma uncertainties on the peak positions, same shape.
        """

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
                    