import gc
import os
import sys
sys.path.append('/home/smarci/python_libs/')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import flap
import flap_w7x_abes

from poloidal_flow import ABESConfig, ABESDataReader

if __name__ == '__main__':

    shotfile = sys.argv[1]

    with open(shotfile, 'r') as f:
        shots = f.read()
        
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin = 0, vmax = 41)
    sm = mpl.cm.ScalarMappable(cmap = cmap, norm = norm)

    for (i, exp_id) in enumerate(shots.split('\n')):

        print(f'Run no. {i}, Exp. ID {exp_id}')

        abes_config = ABESConfig(
            exp_id = exp_id,
            time_range = None,
            channels = np.arange(1, 41),
            bandpass_type = None,
        )

        defl0, defl1 = ABESDataReader(abes_config).read_data()

        apsd = [defl0.apsd(coordinate = 'Time'), defl1.apsd(coordinate = 'Time')]

        # Free up defl0 and defl1 from memory
        del defl0, defl1
        gc.collect()
        
        freqs = [apsd[0].coordinate('Frequency')[0][0], apsd[1].coordinate('Frequency')[0][0]]

        for d in range(2):

            fig, ax = plt.subplots(figsize = (10, 6))
            ax.set_title(f'{exp_id} ABES APSD spectrum, deflection = {d}', fontweight = 'bold')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Spectral intensity [a.u.]')
            ax.set_xscale('log')
            ax.set_yscale('log')

            for ch in range(40):
                color = cmap(norm(ch))
                ax.plot(freqs[d], apsd[d].data[ch], color = color)
            
            ax.grid(which = 'both')
            cbar = fig.colorbar(sm, ax = ax)
            cbar.set_label('Channel number')
            
            plt.savefig(f'plots/apsd/{exp_id}_apsd_defl{d}.pdf')
            plt.close(fig)
        
        del apsd
        flap.delete_data_object('ABES signals', exp_id=exp_id)
        flap.delete_data_object('Beam on', exp_id=exp_id)
        flap.delete_data_object('Beam off', exp_id=exp_id)
        gc.collect()