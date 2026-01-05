"""
Configuration dataclasses for ABES data acquisition and cross-correlation analysis.
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
    time_range : Optional[Tuple[float, float]], optional
        Time range for data acquisition in seconds as (start, end).
        If None, the entire shot duration is used. Default is None.
    channels : List[int], optional
        List of ABES channel numbers to acquire (1-40).
        Default is all 40 channels.
    amplitude_cal : bool, optional
        Whether to apply amplitude calibration to the signals.
        Default is False.
    spatial_cal : bool, optional
        Whether to apply spatial calibration to the channels.
        Default is False.
    bandpass_type : Optional[str], optional
        Type of bandpass filter to apply. Options include 'Elliptic', 'Butterworth',
        or None to skip filtering. Default is 'Elliptic'.
    bandpass_range : Optional[Tuple[float, float]], optional
        Bandpass filter frequency range in Hz as (f_low, f_high).
        Default is (2000, 6000) Hz.
    interpolation_method : str, optional
        Interpolation method for background subtraction. Options include 'linear',
        'cubic', 'quadratic'. Default is 'cubic'.
    pickle_folder : str, optional
        Path to folder for saving/loading pickled data.
        Default is 'pickled_shot_data'.
    """

    exp_id: str
    time_range: Optional[Tuple[float, float]] = None
    channels: List[int] = field(default_factory = lambda: list(range(1, 41)))
    amplitude_cal: bool = False
    spatial_cal: bool = False
    bandpass_type: Optional[str] = 'Elliptic'
    bandpass_range: Optional[Tuple[float, float]] = (2e3, 6e3)  # Hz
    interpolation_method: str = 'cubic'
    pickle_folder: str = 'pickled_shot_data'

@dataclass
class CorrelationConfig:
    """
    Configuration for cross-correlation analysis.

    Parameters
    ----------
    xcorr_fitting_method : str
        Method for extracting time delay from cross-correlation function.
        Options are 'gaussian' (Gaussian curve fitting) or 'cubic_spline'
        (cubic spline interpolation with peak finding).
    xcorr_window : float, optional
        Duration of time window for computing cross-correlation in seconds.
        Default is 0.01 (10 ms).
    xcorr_time_lag_interval: float, optional
        The resulting cross-correlation function is filtered to this time lag interval.
        Default is (-100us, 120us). This should be set in accordance with the chopping
        and deflection frequency of the ABES beam.
    xcorr_resolution : float, optional
        Time resolution for cross-correlation function in seconds.
        Default is 1e-6 (1 microsecond).
    xcorr_interval : int, optional
        Interval parameter for cross-correlation computation.
        Default is 1.
    xcorr_normalize : bool, optional
        Whether to normalize the cross-correlation function.
        Default is True.
    """

    xcorr_fitting_method: str
    xcorr_window: float = 0.01  # seconds
    xcorr_time_lag_interval: Tuple[float, float] = (-1e-4, 1.2e-4)
    xcorr_resolution: float = 1e-6  # seconds
    xcorr_interval: int = 1
    xcorr_normalize: bool = True
