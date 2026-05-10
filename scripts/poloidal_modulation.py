#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:41:20 2025
@author: mive
"""
import os
import copy
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('/home/smarci/python_libs/')

import flap
import flap_w7x_abes as w7x_abes

def get_data(exp_id, signals, deflection=0, plot=False):
           
    for (i, signal) in enumerate(signals):
        dataobject = flap.get_data(
            'W7X_ABES',
            exp_id=exp_id,
            name=f'ABES-{signal}',
            object_name='ABES',
            options={'Amplitude calibration': False, 'Spatial calibration': False}
        )
              
        d_beam_on=flap.get_data(
            'W7X_ABES',
            exp_id=exp_id,
            name='Chopper_time',
            options={'State':{'Chop': 0, 'Defl': deflection},'Start':0,'End':0},\
            object_name='Beam_on'
        )
        
        d = dataobject.slice_data(slicing={'Sample':d_beam_on})
        d = d.slice_data(summing={'Rel. Sample in int(Sample)':'Mean'})
        
        if plot is True:
            tstart=1
            plt.figure(figsize = (10, 8))
            dataobject.slice_data(slicing={"Time":flap.Intervals(tstart,tstart+0.0002)}).plot()
            d.slice_data(slicing={"Time":flap.Intervals(tstart,tstart+0.0002)}).plot(plot_options={"lw":0, "marker":"o"})
            plt.xlim([tstart,tstart+0.0001])
            plt.pause(0.1)
            if signal < 32:
                plt.xlabel("")
            if signal%8!=0:
                plt.ylabel("")
            plt.title(f"ABES-{signal+1}")
            
        if i == 0:
            all_data = np.zeros([len(d.data), len(signals)])
        all_data[:,i] = d.data
    
    #if plot is True:
    #    plt.suptitle(exp_id)
    
    
    dataobject = copy.deepcopy(d)
    dataobject.data = all_data
    channel_coord = flap.Coordinate(name='Channel',
                                 unit='n.a.',
                                 values=signals,
                                 shape=(len(signals)),
                                 mode=flap.CoordinateMode(equidistant=False),
                                 dimension_list=[1])
    dataobject.add_coordinate_object(channel_coord)
    dataobject.shape = dataobject.data.shape
    
    if False:
    
    #adding the spatial data
        try:
            spatcal = w7x_abes.ShotSpatCal(exp_id)
            spatcal.generate_shotdata()
        except FileExistsError: #THis will get triggered if the spatial calibration data was already available
            pass
        spatcal.read()

        for key in spatcal.data.keys():
            if  key != "Channel name":
                new_coord  = copy.deepcopy(channel_coord)
                for index, signal in enumerate(channel_coord.values):
                    new_coord.values[index] = spatcal.data[key][np.where(spatcal.data["Channel name"]==f"ABES-{int(signal)}")][0]
                new_coord.unit.name=key
                new_coord.unit.unit="m"
                dataobject.add_coordinate_object(new_coord)
    
    return dataobject

if __name__ == "__main__":
    w7x_abes.register()
    exp_id = "20250409.046"

    pol0 = get_data(exp_id, [15], deflection=0, plot=True)
    plt.savefig('asd.pdf')
    #pol0 is a typical dataobject, e.g. it can be sliced like pol0.slice_data(slicing={"Time":5}) and all