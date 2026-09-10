"""
ABES data acquisition and preprocessing module.

Turns a W7-X shot into the pair of background-subtracted, filtered
`flap.DataObject`s that `CorrelationAnalysis` consumes, one per deflection
state. The APD system sees plasma light and beam light together, so the beam is
chopped: each deflection state is measured in beam-on periods and the
interleaved beam-off periods give the background to subtract.
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
        Configuration object containing experiment parameters, filter
        settings, and spatial-calibration options.

    Attributes
    ----------
    config : ABESConfig
        The configuration object.
    channel_names : List[str]
        List of channel names in 'ABES-{ch}' format for FLAP data acquisition.
    raw_data : flap.DataObject
        Cached raw signals. Set on first use by ``read_data_raw``.
    on_timings : List[flap.DataObject]
        Cached beam-on chopper timings, indexed by deflection state. Set on
        first use by ``read_timings``.
    off_timings : flap.DataObject
        Cached beam-off chopper timings.

    Notes
    -----
    The raw data and the timings are fetched once and cached on the instance, so
    reading both deflection states costs a single data acquisition.

    Examples
    --------
    >>> config = ABESConfig(exp_id='20250409.046', spatial_cal=False)
    >>> reader = ABESDataReader(config)
    >>> data_defl0, data_defl1 = reader.read_data()
    """
    
    def __init__(self, config: ABESConfig):
        self.config = config
        self.channel_names = [f'ABES-{ch}' for ch in range(1, 41)]

    def read_data(self):
        """
        Read and process ABES data for both deflection states.

        The full pipeline: acquire raw signals, subtract the chopper background,
        and bandpass filter if configured.

        Returns
        -------
        d_defl0 : flap.DataObject
            Background-subtracted and filtered data for deflection state 0,
            shape ``(40 channels, n_samples)``.
        d_defl1 : flap.DataObject
            The same for deflection state 1.

        Notes
        -----
        Thin wrapper over two ``background_subtraction`` calls, which is also
        where the filtering happens. Use that method directly to inspect the
        intermediate beam-on and beam-off signals.

        The two returned objects have slightly different time grids, since the
        two deflection states are measured in interleaved chopper periods, and
        may differ by one sample in length.
        """
        
        _, _, d_defl0 = self.background_subtraction(deflection = 0)
        _, _, d_defl1 = self.background_subtraction(deflection = 1)

        return d_defl0, d_defl1
            
    def read_data_raw(self):
        """
        Read the raw ABES signals of all 40 channels from the W7-X archive.

        Stores the result in ``self.raw_data`` and returns nothing. Called
        automatically by ``background_subtraction`` if the raw data has not been
        loaded yet.

        Notes
        -----
        Uses FLAP's W7X_ABES data source to acquire the signals without any
        processing: beam-on, beam-off and both deflection states are still
        interleaved at this point. Channel names and time range come from the
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
        Read the chopper timings for both deflection states.

        Stores the beam-on intervals in ``self.on_timings``, a list indexed by
        deflection state, and the beam-off intervals in ``self.off_timings``.
        Returns nothing. Called automatically by ``background_subtraction`` if
        the timings have not been loaded yet.

        Notes
        -----
        Three FLAP requests for the 'Chopper_time' signal: beam-on with the
        deflection in state 0 and in state 1 (``Chop: 0, Defl: 0/1``), and the
        common beam-off periods (``Chop: 1``). The returned objects are interval
        descriptors used to slice ``raw_data`` on its 'Sample' coordinate.
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
        Read the radial position of each ABES channel.

        Returns
        -------
        dev_r : numpy.ndarray
            Device R coordinate (major radius) in m for channels 1-40, as a 1D
            array of length 40, ordered by channel number.

        Notes
        -----
        Uses ``flap_w7x_abes.ShotSpatCal``. The calibration shot is
        ``config.spatcal_exp_id`` when set, otherwise ``config.exp_id`` — the
        calibration is not repeated every shot, so pointing several shots at one
        calibration is normal.

        Called by ``background_subtraction`` when ``config.spatial_cal`` is True,
        which attaches the result as a 'Device R' coordinate.
        """
        
        channels = np.arange(1, 41)
        
        if self.config.spatcal_exp_id is not None:
            spatcal_id = self.config.spatcal_exp_id
        else:
            spatcal_id = self.config.exp_id
        
        spatcal = flap_w7x_abes.ShotSpatCal(spatcal_id)
        spatcal.read()
        
        dev_r = np.array([spatcal.data['Device R'][spatcal.data['Channel name'] == f'ABES-{ch}'] for ch in channels]).T[0]
        return dev_r
        
    def apply_bandpass(self, dataobject: flap.DataObject):
        """
        Apply the configured bandpass filter to ABES signal data.

        Parameters
        ----------
        dataobject : flap.DataObject
            Input data to be filtered, typically background-subtracted signals.

        Returns
        -------
        d_bandpass : flap.DataObject
            Filtered data with the same structure as the input.

        Notes
        -----
        Filter design (`bandpass_type`, e.g. 'Butterworth' or 'Elliptic') and
        corner frequencies (`bandpass_range`) come from the configuration; the
        filter runs along the 'Time' coordinate. The band selects the turbulent
        fluctuations the correlation analysis works on, so it has to sit above
        the chopping frequency and below the noise floor.
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
        Subtract the chopper background from one deflection state.

        The APDs see beam light plus plasma background light. The beam-off
        chopper periods measure the background alone, so interpolating them onto
        the beam-on time points and subtracting leaves the beam light.

        Parameters
        ----------
        deflection : int
            State of the poloidal deflection modulation. Must be 0 or 1.

        Returns
        -------
        d_beam_on : flap.DataObject
            Beam-on signal, one sample per chopper period.
        d_beam_off : flap.DataObject
            Beam-off signal (background), one sample per chopper period.
        d_backsub : flap.DataObject
            Background-subtracted signal, bandpass filtered if configured. A
            newly built DataObject with 'Time', 'Channel number' and, when
            ``config.spatial_cal`` is set, 'Device R' coordinates.

        Notes
        -----
        The background subtraction algorithm:
        1. Load raw data and chopper timings (both deflection states) if not
           already cached on the instance
        2. Slice the data into beam-on and beam-off periods
        3. Average the samples within each chopper period, which sets the time
           resolution of the result to one chopper period
        4. Interpolate the beam-off signal onto the beam-on time points
           (``numpy.interp``, linear)
        5. Subtract the interpolated background from the beam-on signal
        6. Apply the bandpass filter if configured

        The time coordinate of the result is rebuilt as equidistant from the
        mean spacing of the beam-on periods, so it is uniform even though the
        chopper periods are not exactly evenly spaced.
        """
        
        if not hasattr(self, 'raw_data'):
            self.read_data_raw()
            
        if not hasattr(self, 'on_timings'):
            self.read_timings()
        
        d_beam_on = self.raw_data.slice_data(slicing={'Sample': self.on_timings[deflection]})
        d_beam_on = d_beam_on.slice_data(summing={'Rel. Sample in int(Sample)': 'Mean'})

        d_beam_off = self.raw_data.slice_data(slicing={'Sample': self.off_timings})
        d_beam_off = d_beam_off.slice_data(summing={'Rel. Sample in int(Sample)': 'Mean'})
        
        backsub_data = d_beam_on.data.copy()
        
        on_time = d_beam_on.coordinate('Time')[0][0]
        off_time = d_beam_off.coordinate('Time')[0][0]
        
        for (i, (on_data, off_data)) in enumerate(zip(d_beam_on.data, d_beam_off.data)):
            backsub_data[i] = on_data - np.interp(on_time, off_time, off_data)
                
        tstart = on_time[0]
        tstep = np.mean(np.diff(on_time))
            
        time_coord = flap.Coordinate(
            name = 'Time',
            unit = 'Second',
            start = tstart,
            step = tstep,
            mode = flap.coordinate.CoordinateMode(equidistant = True),
            dimension_list = [1]
        )
        
                    
        channel_coord = flap.Coordinate(
            name = 'Channel number',
            start = 1,
            step = 1,            
            mode = flap.coordinate.CoordinateMode(equidistant = True),
            dimension_list = [0]
        )
        
        coordinates = [time_coord, channel_coord]
        
        if self.config.spatial_cal:
            
            radial_coord = flap.Coordinate(
                name = 'Device R',
                unit = 'Meter',
                values = self.read_spatial_calibration(),
                mode = flap.coordinate.CoordinateMode(equidistant = False),
                dimension_list = [0]
            )
            coordinates.append(radial_coord)
                
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