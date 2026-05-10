# Process and fit CMOS beam axes

import sys
sys.path.append('/home/smarci/python_libs')

from poloidal_flow.beam_axis.pipeline import CVPipeline
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import flap_w7x_abes

sourcedir = os.path.abspath('cmos')

spatcal = flap_w7x_abes.ShotSpatCalCMOS("20250312.022")
spatcal.read(options={"Shot spatcal dir": "src/poloidal_flow/beam_axis/poldef_spatcal/"})

devx = spatcal.data['Device x']
devy = spatcal.data['Device y']

for (i, imgpath) in enumerate(os.listdir(sourcedir)):
    
    exp_id = imgpath[:-4]
    
    print(f'Fitting {exp_id}')
    
    abspath = os.path.join(sourcedir, imgpath)
    pl = CVPipeline(abspath, (639, 578), 220, 0.015, (devx, devy), 80)
    
    pl.crop_and_threshold()
    pl.find_centroids()
    
    # Too few points
    if len(pl.points) <= 20:
        print('Too few points')
        continue
    
    pl.convert_to_device_coordinates()
    
    try:
        pl.fit_line()
    except:
        print('Failed fit, continuing')
        continue
    
    points = pl.points_phys
    points_x = points[:, 0]
    points_y = points[:, 1]
    
    tau = np.linspace(-0.12, 0.12, 100)
    
    [vx_p, vy_p, x0_p, y0_p] = pl.fit_params
    
    beam_angle = np.rad2deg(np.atan(vx_p/vy_p))
    print(f'Beam angle is {beam_angle:.4f} degs')
    
    if np.abs(beam_angle - 13) >= 4:
        print('Beam angle diverges too much')
        continue

    p = np.array([x0_p, y0_p])
    v = np.array([vx_p, vy_p])

    points_line = p[:, None] + v[:, None] * tau
    
    fig, (ax_img, ax) = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 6))

    plt.suptitle(f'{exp_id} beam point coordinates \n Beam angle = {beam_angle:.4f}$^\\circ$')
    
    ax.set_title('Centroids in device coordinates')
    ax_img.set_title('Processed image with per-row centroids')
    
    ax.set_xlabel('Device x [m]')
    ax.set_ylabel('Device y [m]')
    ax.set_aspect('equal')
    
    output_centr = pl.return_grayscale_img()
    
    ax_img.imshow(output_centr, origin = 'upper')
    ax_img.scatter(pl.points[:, 0], pl.points[:, 1], s = 20)
    
    ax_img.set_xlim(300, 900)
    ax_img.set_ylim(800, 200)
    
    ax_img.set_xlabel('CMOS x [px]')
    ax_img.set_ylabel('CMOS y [px]')
    
    #ax_img.set_xlim(300, 900)
    #ax_img.set_ylim(200, 800)

    ax.scatter(points_x, points_y, s = 10, c = 'green')
    ax.plot(*points_line, c = 'red')
    ax.grid()

    plt.savefig(f'plots/cvoutput/{exp_id}_fit_plot.pdf')
    plt.close(fig)
    
    np.savetxt(f'plots/cvoutput/{exp_id}_fit_data.txt', pl.fit_params)