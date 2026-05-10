#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:03:34 2026

@author: mive
"""

from matplotlib import pyplot as plt

import sys
sys.path.append('/home/smarci/python_libs')

import flap
import flap_w7x_abes

spatcal = flap_w7x_abes.ShotSpatCalCMOS("20250312.022")
spatcal.read(options={"Shot spatcal dir": "./"})
plt.subplot(2,1,1)
plt.title("X coordinate of pixel [m]")
plt.imshow(spatcal.data["Device x"])
plt.colorbar()
plt.subplot(2,1,2)
plt.title("Y coordinate of pixel [m]")
plt.imshow(spatcal.data["Device y"])
plt.colorbar()
plt.tight_layout()

plt.savefig('cmos_pixel_cal.pdf')