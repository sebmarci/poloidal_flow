import sys
sys.path.append('/home/smarci/python_libs')

import flap_w7x_abes

import os
import json
import numpy as np
import shutil

apdcam_path = '/data2/W7-X/APDCAM'
home_path = '/home/smarci/workspace/poloidal_flow/cmos'

with open('reltor.json', 'rb') as f:
    shots = json.load(f)
    
for group in shots:  
    for exp_id in group.keys():
        
        print(f'Reading {exp_id}')
        
        path = os.path.join(apdcam_path, exp_id)
        
        files = os.listdir(path)
        isbmp = ['bmp' in x for x in files]
        
        pics = np.fromiter(filter(lambda x: 'bmp' in x, files), dtype = 'U36')
        pic_idx = np.array([int(x[0:5]) for x in pics])
        
        sorted_idx = np.argsort(pic_idx)
        sorted_pics = pics[sorted_idx]
        
        selected_pic = sorted_pics[20]
        
        try:                                 
            shutil.copy(
                os.path.join(path, selected_pic),
                os.path.join(home_path, f'{exp_id}.bmp')
            )
        except:
            print('failed to copy')
