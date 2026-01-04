"""
ABES data acquisition and preprocessing module.
"""

import flap
import os
import flap_w7x_abes
import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from typing import List, Optional, Tuple
from .config import ABESConfig

class ABESDataReader:
    """
    Reader class for W7-X ABES data acquisition and preprocessing.

    This class handles data acquisition from the W7-X ABES diagnostic system,
    including background subtraction using beam on/off timing, bandpass filtering,
    and data persistence via pickle files.

    Parameters
    ----------
    config : ABESConfig
        Configuration object containing experiment parameters, channel selection,
        filter settings, and file paths.

    Attributes
    ----------
    config : ABESConfig
        The configuration object.
    channel_names : List[str]
        List of channel names in 'ABES-{ch}' format for FLAP data acquisition.

    Examples
    --------
    >>> config = ABESConfig(exp_id='20250409.046', channels=list(range(1, 21)))
    >>> reader = ABESDataReader(config)
    >>> data_defl0, data_defl1 = reader.read_data()
    """
    
    def __init__(self, config: ABESConfig):
        self.config = config
        self.channel_names = [f'ABES-{ch}' for ch in self.config.channels]

    def read_data(self):
        """
        Read and process ABES data for both deflection states.

        Acquires raw data, performs background subtraction, and applies filtering
        for both deflection states (0 and 1) as configured.

        Returns
        -------
        d_defl0 : flap.DataObject
            Background-subtracted and filtered data for deflection state 0.
        d_defl1 : flap.DataObject
            Background-subtracted and filtered data for deflection state 1.

        Notes
        -----
        This method calls `background_subtraction` for each deflection state
        and applies bandpass filtering if configured.
        """
        
        _, _, d_defl0 = self.background_subtraction(deflection = 0)
        _, _, d_defl1 = self.background_subtraction(deflection = 1)

        return d_defl0, d_defl1
    
    def from_pickle(self, defl0_datapath: str, defl1_datapath: str):
        """
        Load previously saved ABES data from pickle files.

        Parameters
        ----------
        defl0_datapath : str
            Filename (not full path) of pickle file for deflection state 0.
        defl1_datapath : str
            Filename (not full path) of pickle file for deflection state 1.

        Returns
        -------
        d_defl0 : flap.DataObject
            Loaded data for deflection state 0, with bandpass filter applied if configured.
        d_defl1 : flap.DataObject
            Loaded data for deflection state 1, with bandpass filter applied if configured.

        Notes
        -----
        Files are loaded from the directory specified in `config.pickle_folder`.
        If `config.bandpass_type` is not None, bandpass filtering is applied
        after loading.
        """
        
        with open(os.path.join(self.config.pickle_folder, defl0_datapath), 'rb') as f:
            d_defl0 = pickle.load(f)
            
        with open(os.path.join(self.config.pickle_folder, defl1_datapath), 'rb') as f:
            d_defl1 = pickle.load(f)
            
        if self.config.bandpass_type is not None:
            d_defl0 = self.apply_bandpass(d_defl0)
            d_defl1 = self.apply_bandpass(d_defl1)
            
        return d_defl0, d_defl1
            
    def read_data_raw(self):
        """
        Read raw ABES signals from W7-X diagnostic system.

        Returns
        -------
        dataobject : flap.DataObject
            Raw ABES signal data for all configured channels and time range.

        Notes
        -----
        Uses FLAP's W7X_ABES data source to acquire raw signals without
        any processing. Channel names and time range are taken from the
        configuration object.
        """
                
        dataobject = flap.get_data(
            'W7X_ABES',
            exp_id = self.config.exp_id,
            name = self.channel_names,
            object_name = 'ABES signals',
            coordinates = {'Time': self.config.time_range}
        )
        
        return dataobject

    def read_timings(self, deflection: int):
        """
        Read beam on/off timing information for background subtraction.

        Parameters
        ----------
        deflection : int
            Deflection state (0 or 1).

        Returns
        -------
        d_on : flap.DataObject
            Timing data for beam-on periods.
        d_off : flap.DataObject
            Timing data for beam-off periods.

        Notes
        -----
        The chopper timing is used to identify when the beam is on (Chop=0)
        vs. off (Chop=1) for the specified deflection state. This is essential
        for background subtraction.
        """
        
        d_on = flap.get_data(
            'W7X_ABES',
            exp_id=self.config.exp_id,
            name='Chopper_time',
            options={'State':{'Chop': 0, 'Defl': deflection}, 'Start':0, 'End':0},
            object_name='Beam on',
            coordinates = {'Time': self.config.time_range}
        )

        d_off = flap.get_data(
            'W7X_ABES',
            exp_id=self.config.exp_id,
            name='Chopper_time',
            options={'State':{'Chop': 1, 'Defl': 0}, 'Start':0, 'End':0},
            object_name='Beam off',
            coordinates = {'Time': self.config.time_range}
        )
        
        return d_on, d_off
    
    def read_spatial_calibration(self):
        """
        Read spatial calibration data for ABES channels.

        Returns
        -------
        dev_r : numpy.ndarray
            Device R coordinates (major radius) for each channel.
        dev_x : numpy.ndarray
            Device X coordinates for each channel.
        dev_y : numpy.ndarray
            Device Y coordinates for each channel.

        Notes
        -----
        Uses the flap_w7x_abes.ShotSpatCal class to retrieve spatial
        calibration information for the configured channels. Coordinates
        are in the device coordinate system.
        """
        
        spatcal = flap_w7x_abes.ShotSpatCal(self.config.exp_id)
        spatcal.read()
        
        dev_r = np.array([spatcal.data['Device R'][spatcal.data['Channel name'] == f'ABES-{ch}'] for ch in self.config.channels])
        dev_x = np.array([spatcal.data['Device x'][spatcal.data['Channel name'] == f'ABES-{ch}'] for ch in self.config.channels])
        dev_y = np.array([spatcal.data['Device y'][spatcal.data['Channel name'] == f'ABES-{ch}'] for ch in self.config.channels])
        
        return dev_r, dev_x, dev_y
        
    def apply_bandpass(self, dataobject: flap.DataObject):
        """
        Apply bandpass filter to ABES signal data.

        Parameters
        ----------
        dataobject : flap.DataObject
            Input data to be filtered.

        Returns
        -------
        d_bandpass : flap.DataObject
            Filtered data with the same structure as input.

        Notes
        -----
        Filter type and frequency range are taken from the configuration.
        Common filter types include 'Elliptic' and 'Butterworth'.
        The filter is applied along the 'Time' coordinate.
        """
        
        d_bandpass = dataobject.filter_data(
                coordinate = 'Time',
                options = {
                    'Type': 'Bandpass',
                    'Design': self.config.bandpass_type,
                    'f_low': self.config.bandpass_range[0],
                    'f_high': self.config.bandpass_range[1]
                }
            )
        
        return d_bandpass
                
    def background_subtraction(self, deflection: int):
        """
        Perform background subtraction on ABES signals.

        Acquires raw signals, separates beam-on and beam-off periods using
        chopper timing, interpolates the beam-off signal, and subtracts it
        from the beam-on signal to remove background light.

        Parameters
        ----------
        deflection : int
            State of poloidal deflection modulation. Must be 0 or 1.

        Returns
        -------
        d_beam_on : flap.DataObject
            Raw beam-on signal data.
        d_beam_off : flap.DataObject
            Raw beam-off signal data (background).
        d_backsub : flap.DataObject
            Background-subtracted signal data, with bandpass filter applied
            if configured.

        Notes
        -----
        The background subtraction algorithm:
        1. Reads raw data and chopper timing for the specified deflection state
        2. Slices data into beam-on and beam-off periods
        3. Averages samples within each chopper period
        4. Interpolates beam-off signal to beam-on time points using cubic
           spline or linear interpolation
        5. Subtracts interpolated background from beam-on signal
        6. Applies bandpass filter if configured

        For single-channel data, uses CubicSpline interpolation. For multi-channel
        data, uses scipy.interpolate.interp1d with cubic interpolation.
        """
        
        dataobject = self.read_data_raw()
        d_on_times, d_off_times = self.read_timings(deflection)

        d_beam_on = dataobject.slice_data(slicing={'Sample': d_on_times})
        d_beam_on = d_beam_on.slice_data(summing={'Rel. Sample in int(Sample)': 'Mean'})

        d_beam_off = dataobject.slice_data(slicing={'Sample': d_off_times})
        d_beam_off = d_beam_off.slice_data(summing={'Rel. Sample in int(Sample)': 'Mean'})
        
        backsub_data = d_beam_on.data
        
        # Edge case if the user requests 1 channel
        if len(self.config.channels) == 1:
                
            on_time = d_beam_on.coordinate('Time')[0]
            off_time = d_beam_off.coordinate('Time')[0]
                
            #off_interp = interp1d(
            #    x = off_time,
            #    y = d_beam_off.data,
            #    kind = 'cubic',
            #    bounds_error = False
            #)
            
            off_interp = CubicSpline(off_time, d_beam_off.data)  
            backsub_data = d_beam_on.data - off_interp(on_time)
        
        # Multiple signals are loaded, data is 2D, channel coordinate is present
        else:
        
            for (i, signame) in enumerate(self.channel_names):
                                
                on_time = d_beam_on.coordinate('Time')[0][0]
                off_time = d_beam_off.coordinate('Time')[0][0]
                    
                off_interp = interp1d(
                    x = off_time,
                    y = d_beam_off.data[i],
                    kind = 'cubic',
                    bounds_error = False
                )
                
                backsub_data[i] = d_beam_on.data[i] - off_interp(on_time)
                
        tstart = on_time[0]
        tstep = np.mean(np.diff(on_time))
            
        time_coord = flap.Coordinate(
            name = 'Time',
            unit = 'Second',
            start = tstart,
            step = tstep,
            mode = flap.coordinate.CoordinateMode(equidistant = True),
            dimension_list = [0]
        )
        
        coordinates = [time_coord]
        
        if len(self.channel_names) != 1:
            
            channel_coord = flap.Coordinate(
            name = 'Channel number',
            values = self.config.channels,
            shape = len(self.config.channels),
            dimension_list = [0],
            mode = flap.coordinate.CoordinateMode(equidistant = False)
            )
            
            time_coord.dimension_list = [1]
            
            coordinates.append(channel_coord)
        
        d_backsub = flap.DataObject(
            exp_id = self.config.exp_id,
            data_title = 'W7-X ABES data',
            data_source = 'W7X_ABES',
            data_unit = flap.Unit(name = 'Signal', unit = 'Volt'),
            data_array = backsub_data,
            coordinates = coordinates
        )
        
        if self.config.bandpass_type is not None:    
            d_backsub = self.apply_bandpass(d_backsub)
            
        return d_beam_on, d_beam_off, d_backsub