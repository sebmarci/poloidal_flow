"""
ABES data acquisition and preprocessing module.
"""

import flap
import flap_w7x_abes
import numpy as np
from typing import List
from .config import ABESConfig

class ABESDataReader:
    """
    Reader class for W7-X ABES data acquisition and preprocessing.

    This class handles data acquisition from the W7-X ABES diagnostic system,
    including background subtraction using beam on/off timing and bandpass filtering.

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
            
    def read_data_raw(self):
        """
        Read raw ABES signals from W7-X diagnostic system.

        Stores the result in ``self.raw_data``. Called automatically by
        ``background_subtraction`` if raw data has not been loaded yet.

        Notes
        -----
        Uses FLAP's W7X_ABES data source to acquire raw signals without
        any processing. Channel names and time range are taken from the
        configuration object.
        """
                
        raw_data = flap.get_data(
            'W7X_ABES',
            exp_id = self.config.exp_id,
            name = self.channel_names,
            object_name = 'ABES signals',
            coordinates = {'Time': self.config.time_range}
        )
        
        self.raw_data = raw_data

    def read_timings(self):
        """
        Read beam on/off timing information for both deflection states.

        Stores results in ``self.on_timings`` (list indexed by deflection state)
        and ``self.off_timings``. Called automatically by ``background_subtraction``
        if timings have not been loaded yet.

        Notes
        -----
        Fetches chopper timings for both deflection states (Defl=0 and Defl=1)
        and the beam-off periods (Chop=1) in a single call.
        """
        
        defl0_timings = flap.get_data(
            'W7X_ABES',
            exp_id=self.config.exp_id,
            name='Chopper_time',
            options={'State':{'Chop': 0, 'Defl': 0}, 'Start':0, 'End':0},
            object_name='Beam on',
            coordinates = {'Time': self.config.time_range}
        )
        
        defl1_timings = flap.get_data(
            'W7X_ABES',
            exp_id=self.config.exp_id,
            name='Chopper_time',
            options={'State':{'Chop': 0, 'Defl': 1}, 'Start':0, 'End':0},
            object_name='Beam on',
            coordinates = {'Time': self.config.time_range}
        )

        off_timings = flap.get_data(
            'W7X_ABES',
            exp_id=self.config.exp_id,
            name='Chopper_time',
            options={'State':{'Chop': 1, 'Defl': 0}, 'Start':0, 'End':0},
            object_name='Beam off',
            coordinates = {'Time': self.config.time_range}
        )
        
        self.on_timings = [defl0_timings, defl1_timings]
        self.off_timings = off_timings
         
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
        1. Loads raw data and chopper timings (both deflection states) if not
           already cached on the instance
        2. Slices data into beam-on and beam-off periods
        3. Averages samples within each chopper period
        4. Interpolates beam-off signal to beam-on time points using linear
           interpolation (``numpy.interp``)
        5. Subtracts interpolated background from beam-on signal
        6. Applies bandpass filter if configured
        """
        
        if not hasattr(self, 'raw_data'):
            self.read_data_raw()
            
        if (not hasattr(self, 'on_timings')):
            self.read_timings()
        
        d_beam_on = self.raw_data.slice_data(slicing={'Sample': self.on_timings[deflection]})
        d_beam_on = d_beam_on.slice_data(summing={'Rel. Sample in int(Sample)': 'Mean'})

        d_beam_off = self.raw_data.slice_data(slicing={'Sample': self.off_timings})
        d_beam_off = d_beam_off.slice_data(summing={'Rel. Sample in int(Sample)': 'Mean'})
        
        backsub_data = d_beam_on.data.copy()
        
        # Edge case if the user requests 1 channel
        if len(self.config.channels) == 1:
                
            on_time = d_beam_on.coordinate('Time')[0]
            off_time = d_beam_off.coordinate('Time')[0]
            
            backsub_data = d_beam_on.data - np.interp(on_time, off_time, d_beam_off.data)
        
        # Multiple signals are loaded, data is 2D, channel coordinate is present
        else:
        
            for (i, (on_data, off_data)) in enumerate(zip(d_beam_on.data, d_beam_off.data)):
                                
                on_time = d_beam_on.coordinate('Time')[0][0]
                off_time = d_beam_off.coordinate('Time')[0][0]
                
                backsub_data[i] = on_data - np.interp(on_time, off_time, off_data)
                
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