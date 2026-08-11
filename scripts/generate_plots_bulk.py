import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
import flap_w7x_abes
import flap

SPATCAL_EXPID = '20250409.046'
CORR_THRESHOLD = 0.35

# Read spatial calibration data
print(f'Reading spatcal data from {SPATCAL_EXPID}')
spatcal = flap_w7x_abes.ShotSpatCal(SPATCAL_EXPID)
spatcal.read()
dev_r = np.array([
    spatcal.data['Device R'][spatcal.data['Channel name'] == f'ABES-{ch}'] for ch in np.arange(1, 41)
]).T[0]

mpl.rcParams.update({
    'font.size': 12
})

for exp_id in os.listdir('processed_data'):
    
    print(f'Processing {exp_id}')
    
    if os.path.isdir(f'plots/{exp_id}'):
        print('Folder already exists')
        continue
    
    os.mkdir(f'plots/{exp_id}')
    
    with h5py.File(f'processed_data/{exp_id}/{exp_id}.h5', 'r') as f:
    
        udefl = f['ccf_data'].attrs['poloidal_deflection_voltage']
        deltay = f['ccf_data'].attrs['poloidal_separation']
        twindow = f['ccf_data'].attrs['ccf_window']
        intervaln = f['ccf_data'].attrs['ccf_interval_n']
        
        times = f['ccf_data/time_range'][:]
        channels = f['ccf_data/channels'][:]
        
        tau = f['ccf_data/ccf_max_time_lags'][:]
        corrs = f['ccf_data/ccf_correlations'][:]
        vpol = f['ccf_data/poloidal_velocity'][:]
        vpol_err = f['ccf_data/poloidal_velocity_err'][:]
        
    print(f'U_defl = {udefl} V')
    print(f'Poloidal sep = {deltay} mm')
    print(f't_window = {twindow} s')
    print(f'Interval n = {intervaln}')
    
    vpol_filtered = vpol.copy()
    vpol_err_filtered = vpol_err.copy()
    vpol_filtered[corrs < CORR_THRESHOLD] = np.nan
    vpol_err_filtered[corrs < CORR_THRESHOLD] = np.nan
    
    fig, ax = plt.subplots(figsize = (24, 8), nrows = 1, ncols = 2)
    plt.suptitle(f'{exp_id} CCF maximum time lags, $t_{{window}} = {twindow*1e3}$ ms, $U_{{defl}} = {udefl} V$', fontsize = 20)

    ax[0].set_title('Time lags')
    ax[0].set_xlabel('Device R {m}')
    ax[0].set_ylabel('Time [s]')
    pc1 = ax[0].pcolormesh(dev_r, times, tau, cmap = 'bwr', vmin = -20, vmax = 20, shading = 'nearest')
    cbar1 = fig.colorbar(pc1, ax = ax[0])
    cbar1.set_label('$\\tau_{max}$ [$\\mu$s]')

    ax[1].set_title('Correlation at time lag')
    ax[1].set_xlabel('Device R [m]')
    ax[1].set_ylabel('Time [s]')
    pc2 = ax[1].pcolormesh(dev_r, times, corrs, cmap = 'viridis', shading = 'nearest')
    cbar2 = fig.colorbar(pc2, ax = ax[1])
    cbar2.set_label('$C_{01}(\\tau_{max})$')

    plt.savefig(f'plots/{exp_id}/{exp_id}_ccf_corrs.pdf')
    plt.close(fig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 8))
    plt.suptitle(f'{exp_id} poloidal velocity profile and error', fontsize = 18)

    ax1.set_title('Velocity')
    ax1.set_xlabel('Device R [m]')
    ax1.set_ylabel('Time [s]')
    pc1 = ax1.pcolormesh(dev_r, times, vpol_filtered, cmap = 'bwr', vmin = -50, vmax = 50, shading = 'nearest')
    cbar1 = fig.colorbar(pc1, ax = ax1)
    cbar1.set_label('$v_{pol}$ [km/s]')

    ax2.set_title('Error of velocity')
    ax2.set_xlabel('Device R [m]')
    ax2.set_ylabel('Time [s]')
    pc2 = ax2.pcolormesh(dev_r, times, vpol_err_filtered, cmap = 'viridis', vmin = 0, vmax = 20, shading = 'nearest')
    cbar2 = fig.colorbar(pc2, ax = ax2)
    cbar2.set_label('$\\Delta v_{pol}$ [km/s]')

    plt.savefig(f'plots/{exp_id}/{exp_id}_vpol.pdf')
    plt.close(fig)
    
    # Plot with main parameters
        
    try:
        i_tor = flap.load(f'plasma_param_data/{exp_id}/{exp_id}_itor.pkl')
        p_ecrh = flap.load(f'plasma_param_data/{exp_id}/{exp_id}_ecrh.pkl')
        w_dia = flap.load(f'plasma_param_data/{exp_id}/{exp_id}_wdia.pkl')
        n_int = flap.load(f'plasma_param_data/{exp_id}/{exp_id}_lineint.pkl')
    except:
        print('Main parameters not found')
        continue

    fig, axes = plt.subplots(
        nrows = 5,
        ncols = 1,
        figsize = (9, 12),
        gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]},
        sharex = True
    )
    
    plt.suptitle(f'{exp_id} max time lags')

    pcm = axes[0].pcolormesh(times, dev_r, tau.T, shading = 'nearest', vmin = -15, vmax = 15, cmap = 'bwr')
    axes[0].set_ylabel('Device R [m]')

    axes[1].plot(p_ecrh.coordinate('Time')[0], p_ecrh.data / 1e3)
    axes[1].set_ylabel('$P_{ECRH}$ [MW]')

    axes[2].plot(i_tor.coordinate('Time')[0], i_tor.data / 1e3, c = 'tab:green')
    axes[2].set_ylabel('$I_{tor}$ [kA]')

    axes[3].plot(n_int.coordinate('Time')[0], n_int.data, c = 'tab:red')
    axes[3].set_ylabel('$\\int n \\: dl$  [$10^{19} m^{-3}$]')

    axes[4].plot(w_dia.coordinate('Time')[0], w_dia.data, c = 'tab:orange')
    axes[4].set_ylabel('$W_{dia}$ [kJ]')

    axes[4].set_xlabel('Time [s]')

    axes[4].set_xlim(0, 10)

    for ax in axes: ax.grid()

    caxes = [make_axes_locatable(ax).append_axes('right', size = '2%', pad = 0.1) for ax in axes]
    for cax in caxes[1:]: cax.set_axis_off()

    cbar = fig.colorbar(pcm, cax = caxes[0])
    cbar.set_label('$\\tau_{max}$ [$\\mu$s]')
    
    plt.savefig(f'plots/{exp_id}/{exp_id}_taumax_params.pdf')
    plt.close(fig)
    
    fig, axes = plt.subplots(
            nrows = 5,
            ncols = 1,
            figsize = (9, 12),
            gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]},
            sharex = True
    )
    
    plt.suptitle(f'{exp_id} correlations')
    
    pcm = axes[0].pcolormesh(times, dev_r, corrs.T, shading = 'nearest', cmap = 'viridis')
    axes[0].set_ylabel('Device R [m]')
    
    axes[1].plot(p_ecrh.coordinate('Time')[0], p_ecrh.data / 1e3)
    axes[1].set_ylabel('$P_{ECRH}$ [MW]')
    
    axes[2].plot(i_tor.coordinate('Time')[0], i_tor.data / 1e3, c = 'tab:green')
    axes[2].set_ylabel('$I_{tor}$ [kA]')
    
    axes[3].plot(n_int.coordinate('Time')[0], n_int.data, c = 'tab:red')
    axes[3].set_ylabel('$\\int n \\: dl$  [$10^{19} m^{-3}$]')
    
    axes[4].plot(w_dia.coordinate('Time')[0], w_dia.data, c = 'tab:orange')
    axes[4].set_ylabel('$W_{dia}$ [kJ]')
    
    axes[4].set_xlabel('Time [s]')
    
    axes[4].set_xlim(0, 10)
    
    for ax in axes: ax.grid()
    
    caxes = [make_axes_locatable(ax).append_axes('right', size = '2%', pad = 0.1) for ax in axes]
    for cax in caxes[1:]: cax.set_axis_off()
    
    cbar = fig.colorbar(pcm, cax = caxes[0])
    cbar.set_label('$\\tau_{max}$ [$\\mu$s]')
        
    plt.savefig(f'plots/{exp_id}/{exp_id}_corr_params.pdf')
    plt.close(fig)
    
    fig, axes = plt.subplots(
                nrows = 5,
                ncols = 1,
                figsize = (9, 12),
                gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]},
                sharex = True
    )
        
    plt.suptitle(f'{exp_id} poloidal velocities')
        
    pcm = axes[0].pcolormesh(times, dev_r, vpol.T, shading = 'nearest', vmin = -15, vmax = 15, cmap = 'bwr')
    axes[0].set_ylabel('Device R [m]')
        
    axes[1].plot(p_ecrh.coordinate('Time')[0], p_ecrh.data / 1e3)
    axes[1].set_ylabel('$P_{ECRH}$ [MW]')
        
    axes[2].plot(i_tor.coordinate('Time')[0], i_tor.data / 1e3, c = 'tab:green')
    axes[2].set_ylabel('$I_{tor}$ [kA]')
        
    axes[3].plot(n_int.coordinate('Time')[0], n_int.data, c = 'tab:red')
    axes[3].set_ylabel('$\\int n \\: dl$  [$10^{19} m^{-3}$]')
        
    axes[4].plot(w_dia.coordinate('Time')[0], w_dia.data, c = 'tab:orange')
    axes[4].set_ylabel('$W_{dia}$ [kJ]')
        
    axes[4].set_xlabel('Time [s]')
        
    axes[4].set_xlim(0, 10)
        
    for ax in axes: ax.grid()
        
    caxes = [make_axes_locatable(ax).append_axes('right', size = '2%', pad = 0.1) for ax in axes]
    for cax in caxes[1:]: cax.set_axis_off()
        
    cbar = fig.colorbar(pcm, cax = caxes[0])
    cbar.set_label('$\\tau_{max}$ [$\\mu$s]')
            
    plt.savefig(f'plots/{exp_id}/{exp_id}_vpol_params.pdf')
    plt.close(fig)
    
    # Read light profile data
    lightprof_path = ''
    electron_dens_path = ''

    flap_recon_path = os.path.abspath(f'/data2/W7-X/processed_data/APDCAM/flap_recon/{exp_id}')

    try:
        for file in os.listdir(flap_recon_path):
            if re.search(r'light_ds_orig', file):
                lightprof_path = file
            elif re.search(r'dens', file):
                electron_dens_path = file
    except FileNotFoundError:
        print('No recon folder found for this')
        continue
            
    print('Matched recon data files:')
    print(lightprof_path)
    print(electron_dens_path)

    lightprof = flap.load(os.path.join(flap_recon_path, lightprof_path))
    electron_dens = flap.load(os.path.join(flap_recon_path, electron_dens_path))

    lightprof_avg = lightprof.slice_data(
        slicing = {'Time': flap.Intervals(0, 10)},
        summing = {'Time': 'Mean'}
    )

    electron_dens_avg = electron_dens.slice_data(
        slicing = {'Time': flap.Intervals(0, 10)},
        summing = {'Time': 'Mean'}
    )
    
    lightprof_max = lightprof_avg.coordinate('Device R')[0][np.argmax(lightprof_avg.data)]
    
    fig, ax = plt.subplots(
        figsize = (10, 8),
        nrows = 3, ncols = 2,
        sharex = True,
        gridspec_kw = {'height_ratios':[2, 1, 1]},
        constrained_layout = True
    )

    plt.suptitle(f'{exp_id} CCF maximum time lags with profiles')

    ax[0, 0].set_title('Time lags')
    ax[2, 0].set_xlabel('Device R [m]')
    ax[2, 1].set_xlabel('Device R [m]')
    ax[0, 0].set_ylabel('Time [s]')
    pc1 = ax[0, 0].pcolormesh(dev_r, times, tau, cmap = 'bwr', vmin = -20, vmax = 20, shading = 'nearest')
    cbar1 = fig.colorbar(pc1, ax = ax[0, 0])
    cbar1.set_label('$\\tau_{max}$ [$\\mu$s]')

    ax[0, 1].set_title('Correlation at time lag')
    ax[0, 1].set_ylabel('Time [s]')
    pc2 = ax[0, 1].pcolormesh(dev_r, times, corrs, cmap = 'viridis', shading = 'nearest')
    cbar2 = fig.colorbar(pc2, ax = ax[0, 1])
    cbar2.set_label('$C_{01}(\\tau_{max})$')

    ax[1, 0].set_title('Time-averaged light profile [0, 10] s')
    ax[1, 0].set_ylabel('Light intensity')
    ax[1, 1].set_ylabel('Light intensity')

    ax[1, 1].plot(lightprof_avg.coordinate('Device R')[0], lightprof_avg.data)
    ax[1, 0].plot(lightprof_avg.coordinate('Device R')[0], lightprof_avg.data)

    ax[2, 0].set_title('Time-averaged electron density profile [0, 10] s')
    ax[2, 0].set_ylabel('$\\langle n_e \\rangle$ [$10^{19}$ m$^{-3}$]')
    ax[2, 1].set_ylabel('$\\langle n_e \\rangle$ [$10^{19}$ m$^{-3}$]')

    ax[2, 0].plot(electron_dens_avg.coordinate('Device R')[0], electron_dens_avg.data, c = 'r')
    ax[2, 1].plot(electron_dens_avg.coordinate('Device R')[0], electron_dens_avg.data, c = 'r')

    plt.savefig(f'plots/{exp_id}/{exp_id}_ccf_light_dens_spline.pdf')
    plt.close(fig)