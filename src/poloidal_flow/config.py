"""
Configuration dataclasses for ABES data acquisition and cross-correlation analysis.

`ABESConfig` is consumed by `ABESDataReader`, `CorrelationConfig` by
`CorrelationAnalysis` (and through it by `CCFPlotter`). One instance of each is
the single source of truth for a run and is what gets recorded alongside the
results.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List

@dataclass
class ABESConfig:
    """
    Configuration for ABES data acquisition and preprocessing.

    Parameters
    ----------
    exp_id : str
        Experiment ID (shot number).
    spatial_cal : bool, optional
        If True, attach a 'Device R' coordinate to the background-subtracted
        DataObject using ``ABESDataReader.read_spatial_calibration``.
        Default is False.
    spatcal_exp_id : Optional[str], optional
        Experiment ID to use when looking up the spatial calibration.
        If None, ``exp_id`` is used. Default is None.
    time_range : Optional[Tuple[float, float]], optional
        Time range for data acquisition in seconds as (start, end).
        If None, the entire shot duration is used. Default is None.
    bandpass_type : Optional[str], optional
        Type of bandpass filter to apply. Options include 'Elliptic', 'Butterworth',
        or None to skip filtering. Default is None.
    bandpass_range : Optional[Tuple[float, float]], optional
        Bandpass filter frequency range in Hz as (f_low, f_high). Ignored when
        `bandpass_type` is None. Default is (2000, 10000) Hz.

    Examples
    --------
    >>> config = ABESConfig(
    ...     exp_id='20250409.046',
    ...     time_range=(0, 10),
    ...     bandpass_type='Butterworth',
    ...     bandpass_range=(2e3, 10e3),
    ... )
    """

    exp_id: str
    spatial_cal: bool = False
    spatcal_exp_id: str = None
    time_range: Optional[Tuple[float, float]] = None
    bandpass_type: Optional[str] = None
    bandpass_range: Optional[Tuple[float, float]] = (2e3, 10e3)  # Hz

@dataclass
class CorrelationConfig:
    """
    Configuration for cross-correlation analysis.

    Parameters
    ----------
    xcorr_fitting_method : str
        Method for extracting the time delay from the cross-correlation
        function. Use 'parabola', a parabola fit to the 5 samples around the
        peak, which is the only method that yields an uncertainty and the only
        one accepted by the ``get_max_time_lag_*`` methods. 'gaussian' and
        'cubic_spline' are deprecated and usable for plotting only.
    xcorr_window : float, optional
        Duration of the time window for one cross-correlation, in seconds.
        Default is 0.05 (50 ms); production runs use 1 s.
    xcorr_time_lag_interval : Tuple[float, float], optional
        The cross-correlation function is restricted to this time lag interval,
        in seconds. Default is (-100 us, 120 us). Set it in accordance with the
        chopping and deflection frequency of the ABES beam: a peak reaching the
        edge of the interval cannot be fitted.
    apdcam_path : str, optional
        Directory holding the APDCAM XML configurations, from which the chopper
        deflection voltage is read. Default is '/data2/W7-X/APDCAM'.
    xcorr_resolution : float, optional
        Time resolution of the cross-correlation function in seconds.
        Default is None, meaning the highest resolution the data allows.
    xcorr_interval : int, optional
        Number of subintervals the window is split into. FLAP derives the CCF
        error from the scatter across them, so this **must be > 1**: with the
        default of 1 the errors are all zero and the weighted parabola fit
        silently returns its initial guess. Production runs use 100-150.
    xcorr_normalize : bool, optional
        Whether to normalize the correlation functions to 1 at zero lag.
        Default is True and it **must stay True** for the elliptical model,
        whose defining equation ``ACF(tau_0) = CCF_max`` is otherwise
        meaningless.

    Examples
    --------
    >>> config = CorrelationConfig(
    ...     xcorr_fitting_method='parabola',
    ...     xcorr_window=1.0,
    ...     xcorr_interval=150,
    ... )
    """
    
    xcorr_fitting_method: str
    xcorr_window: float = 0.05  # seconds
    xcorr_time_lag_interval: Tuple[float, float] = (-1e-4, 1.2e-4)
    apdcam_path: str = '/data2/W7-X/APDCAM'
    xcorr_resolution: float = None  # seconds (None is highest resolution by default)
    xcorr_interval: int = 1
    xcorr_normalize: bool = True