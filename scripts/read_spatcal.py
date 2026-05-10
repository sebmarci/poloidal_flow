#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 10:28:07 2025

@author: mive
"""

import sys
sys.path.append('/home/smarci/python_libs')

from matplotlib import pyplot as plt
import numpy as np

import flap_w7x_abes


if __name__ == "__main__":
    exp_ID = '20250409.046'
    r_lcfs = 6.232133974825017 #m
    spatcal = flap_w7x_abes.ShotSpatCal(exp_ID)
    spatcal.read()# reading the file in the folder
    
    xylocations = dict()
    rlocations = dict()
    index = 0
    for channel in spatcal.data['Channel name']:
        if "ABES" in channel: #Changing names like ABES-1, ABES-2 to ABES-01, ABES-02, etc.
            channel = f"ABES-{channel.split("-")[1].zfill(2)}"
        xylocations[channel] = [float(spatcal.data['Device x'][index]), float(spatcal.data['Device y'][index])]
        rlocations[channel] = [float(spatcal.data['Device R'][index])]
        index += 1
    
    #Plotting the X and Y coordinates and the Major radii of the channel locations 
    fig = plt.figure(figsize=[10,12])
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    keys = sorted(xylocations.keys())
    colorvalues = np.linspace(1,0,len(keys))
    colorvect = [(colorvalue, abs(2*colorvalue-1),0) for colorvalue in colorvalues]
    print(f"Channel\tXY coord [m]\tMajor radius [m]")
    for index, key in enumerate(keys):
        print(f"{key}\t{xylocations[key]}\t{rlocations[key][0]}")
        ax1.scatter(xylocations[key][0], xylocations[key][1], color=colorvect[index])
        ax1.text(xylocations[key][0]+0.002, xylocations[key][1]-0.002, key)
        
        if "ABES" in key:
            ax2.scatter(index+1, rlocations[key][0], color=colorvect[index])
            ax2.text(index+1.5, rlocations[key][0], key)
        
    #Plotting the LCFS location on the second plot
    ax2.hlines(r_lcfs,0,41, color="tab:blue")
    ax2.text(35, r_lcfs+0.002,"LCFS", color="tab:blue")
    ax1.set_xlabel("X coordinate [m]")
    ax1.set_ylabel("Y coordinate [m]")
    ax2.set_xlabel("Channel index")
    ax2.set_ylabel("Major radius [m]")
    fig.tight_layout()
    
    plt.savefig('spatcal.pdf')
