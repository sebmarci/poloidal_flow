import sys
import os

sys.path.append('/home/smarci/python_libs')

import flap
import flap_w7x_abes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 12
})

exp_id = '20250409.046'

spatcal = flap_w7x_abes.ShotSpatCal('20250409.046')
spatcal.read()
device_r = np.array([
    spatcal.data['Device R'][spatcal.data['Channel name'] == f'ABES-{ch}'] for ch in np.arange(1, 41)
])[:, 0]

r_lcfs = 6.232133974825017 # m

savepath = os.path.abspath('plots/cpsd/20250409.046/ecrh_high')

ref_chs = np.arange(5, 20)
channels = np.arange(1, 41)

defl0 = flap.load(f'pickled_shot_data/{exp_id}_raw_defl0.pkl')
defl1 = flap.load(f'pickled_shot_data/{exp_id}_raw_defl1.pkl')

defl0 = defl0.slice_data(
    slicing = {'Time': flap.Intervals(6, 7)}
)

defl1 = defl1.slice_data(
    slicing = {'Time': flap.Intervals(6, 7)}
)

lightprof = flap.load(
    '/data2/W7-X/processed_data/APDCAM/flap_recon/20250409.046/20250409.046_1.5500000000001624e-05-10.6199955_light_orig_1774945808861.hdf5'
)

lightprof_avg = lightprof.slice_data(
    slicing = {'Time': flap.Intervals(6, 7)},
    summing = {'Time': 'Mean'}
)

lightprof_max_idx = np.argmax(lightprof_avg.data)

print('Calculating CPSD')

cpsd0 = defl0.cpsd(
    ref=defl0,
    coordinate='Time',
    options={
        'Range': [100, 1e4],
        'Interval_n': 10,
        'Normalize': True,
        'Wavenumber': False,
        'Resolution': None,
        'Trend removal': None
    }
)

cpsd1 = defl1.cpsd(
    ref=defl1,
    coordinate='Time',
    options={
        'Range': [100, 1e4],
        'Interval_n': 10,
        'Normalize': True,
        'Wavenumber': False,
        'Resolution': None,
        'Trend removal': None
    }
)

for rch in ref_chs:
    
    print(f'Plotting ref CH {rch}')
    
    cpsd0_slice = cpsd0.slice_data(
        slicing = {'Channel number (Ref)': rch}
    )
    
    cpsd1_slice = cpsd1.slice_data(
        slicing = {'Channel number (Ref)': rch}
    )
    
    freq0 = cpsd0_slice.coordinate('Frequency')[0][0]
    freq1 = cpsd1_slice.coordinate('Frequency')[0][0]
    
    fig, ax = plt.subplots(
        nrows = 2, ncols = 2,
        figsize=(15, 6),
        gridspec_kw = {'height_ratios': [2, 1]},
        sharex = True,
        constrained_layout = True
    )
    
    ax[1, 0].set_xlabel('Device R [m]')
    ax[1, 1].set_xlabel('Device R [m]')
    
    ax[0, 0].set_ylabel('Frequency [Hz]')
    
    ax[0, 0].set_ylim([100, 10000])
    ax[0, 1].set_ylim([100, 10000])

    ax[0, 0].set_yscale('log')
    ax[0, 1].set_yscale('log')

    pcm_abs = ax[0, 0].pcolormesh(
        device_r,
        freq0,
        np.abs(cpsd0_slice.data).T,
        shading='nearest',
        cmap='plasma',
    )
    
    pcm_phase = ax[0, 1].pcolormesh(
        device_r,
        freq0,
        np.angle(cpsd0_slice.data, deg = True).T,
        shading='nearest',
        vmin = -180, vmax = 180,
        cmap='twilight',
    )
    
    ax[1, 0].plot(lightprof_avg.coordinate('Device R')[0], lightprof_avg.data)
    ax[1, 1].plot(lightprof_avg.coordinate('Device R')[0], lightprof_avg.data)
    
    ax[0, 0].axvline(lightprof_avg.coordinate('Device R')[0][lightprof_max_idx], c = 'green', ls = '--', lw = 1.5, label = 'Light profile maximum')
    ax[1, 0].axvline(lightprof_avg.coordinate('Device R')[0][lightprof_max_idx], c = 'green', ls = '--', lw = 1.5)
    ax[1, 1].axvline(lightprof_avg.coordinate('Device R')[0][lightprof_max_idx], c = 'green', ls = '--', lw = 1.5)
    ax[0, 1].axvline(lightprof_avg.coordinate('Device R')[0][lightprof_max_idx], c = 'green', ls = '--', lw = 1.5)
    
    ax[0, 0].axvline(r_lcfs, c = 'magenta', ls = '--', lw = 1.5, label = 'LCFS')
    ax[0, 1].axvline(r_lcfs, c = 'magenta', ls = '--', lw = 1.5)
    ax[1, 0].axvline(r_lcfs, c = 'magenta', ls = '--', lw = 1.5)
    ax[1, 1].axvline(r_lcfs, c = 'magenta', ls = '--', lw = 1.5)

    cbar = fig.colorbar(pcm_abs, ax=ax[0, 0])
    cbar.set_label('Abs. value of cross-coherence')

    cbar = fig.colorbar(pcm_phase, ax=ax[0, 1])
    cbar.set_label('Phase of cross-coherence (deg)')
    
    ax[0, 0].legend()

    plt.suptitle(f'{exp_id} CPSD, deflection = 0, reference: CH {rch}, t $\\in$ [6, 7] s', fontsize = 16)
    plt.savefig(os.path.join(savepath, 'defl0', f'ref{rch}.pdf'))
    plt.close(fig)
    
    fig, ax = plt.subplots(
        nrows = 2, ncols = 2,
        figsize=(15, 6),
        gridspec_kw = {'height_ratios': [2, 1]},
        sharex = True,
        constrained_layout = True
    )
    
    ax[1, 0].set_xlabel('Device R [m]')
    ax[1, 1].set_xlabel('Device R [m]')
    
    ax[0, 0].set_ylabel('Frequency [Hz]')
    
    ax[0, 0].set_ylim([100, 10000])
    ax[0, 1].set_ylim([100, 10000])

    ax[0, 0].set_yscale('log')
    ax[0, 1].set_yscale('log')

    pcm_abs = ax[0, 0].pcolormesh(
        device_r,
        freq1,
        np.abs(cpsd1_slice.data).T,
        shading='nearest',
        cmap='plasma',
    )
    
    pcm_phase = ax[0, 1].pcolormesh(
        device_r,
        freq1,
        np.angle(cpsd1_slice.data, deg = True).T,
        shading='nearest',
        vmin = -180, vmax = 180,
        cmap='twilight',
    )
    
    ax[1, 0].plot(lightprof_avg.coordinate('Device R')[0], lightprof_avg.data)
    ax[1, 1].plot(lightprof_avg.coordinate('Device R')[0], lightprof_avg.data)
    
    ax[0, 0].axvline(lightprof_avg.coordinate('Device R')[0][lightprof_max_idx], c = 'green', ls = '--', lw = 1.5, label = 'Light profile maximum')
    ax[1, 0].axvline(lightprof_avg.coordinate('Device R')[0][lightprof_max_idx], c = 'green', ls = '--', lw = 1.5)
    ax[1, 1].axvline(lightprof_avg.coordinate('Device R')[0][lightprof_max_idx], c = 'green', ls = '--', lw = 1.5)
    ax[0, 1].axvline(lightprof_avg.coordinate('Device R')[0][lightprof_max_idx], c = 'green', ls = '--', lw = 1.5)

    ax[0, 0].axvline(r_lcfs, c = 'magenta', ls = '--', lw = 1.5, label = 'LCFS')
    ax[0, 1].axvline(r_lcfs, c = 'magenta', ls = '--', lw = 1.5)
    ax[1, 0].axvline(r_lcfs, c = 'magenta', ls = '--', lw = 1.5)
    ax[1, 1].axvline(r_lcfs, c = 'magenta', ls = '--', lw = 1.5)

    cbar = fig.colorbar(pcm_abs, ax=ax[0, 0])
    cbar.set_label('Abs. value of cross-coherence')

    cbar = fig.colorbar(pcm_phase, ax=ax[0, 1])
    cbar.set_label('Phase of cross-coherence (deg)')
    
    ax[0, 0].legend()

    plt.suptitle(f'{exp_id} CPSD, deflection = 1, reference: CH {rch}, t $\\in$ [6, 7] s', fontsize = 16)
    plt.savefig(os.path.join(savepath, 'defl1', f'ref{rch}.pdf'))
    plt.close(fig)