import numpy as np
from microscope import Microscope
from fishbowl import Fishbowl
from scipy.constants import epsilon_0, mu_0, c
import json, os
from misc_functions import *

if __name__ == '__main__':
    Mag = 60

    wl = 690e-9
    for wl in np.linspace(420,690,10)*1e-9:
        freq = c/wl

        n_obj = 1.33
        n_sub = n_obj
        n_cam = 1
        n = [n_obj,n_sub,n_cam]

        mur_obj = 1
        mur_sub = 1
        mur_cam = 1
        mur = [mur_obj,mur_sub,mur_cam]

        epsr_obj = n_obj**2/mur_obj
        epsr_sub = n_sub**2/mur_sub
        epsr_cam = n_cam**2/mur_cam
        epsr = [epsr_obj,epsr_sub,epsr_cam]

        k_0 = 2*np.pi*freq*np.sqrt(epsilon_0*mu_0)

        z_Interface_sub = -30e-9

        FoV = 10
        NA = 1.2
        camera_size = 6
        voxel_size = 60e-9/690e-9

        M_timepoints = 100
        N_sensors = 100
        NA = 1.2
        f_obj = 5e-2
        f_cam = f_obj*(Mag*n_cam/n_obj)
        f = [f_obj,f_cam]
        off = 0

        dir = 'single_variable/microscope/wl'
        dipoles = np.array([[-0.0046*wl,0*wl,0*wl],[0.0046*wl,0*wl,0*wl]])

        if os.path.isfile(dir+'/{}_data_microscope.json'.format(wl/1e-9)):
            continue
        print(wl/1e-9)

        mic = Microscope(Mag,N_sensors,wl,n,mur,epsr,k_0,f,NA,z_Interface_sub,dipoles,voxel_size,M_timepoints)
        mic.create_image_stack()
        mic.find_resolution_limit()
        print(mic.resolution_limit)
        mic.save_info(dir,wl/1e-9)
        # exit()

        # FoV = 0.01
        # fib = Fishbowl(N_sensors,f_obj,wl,n_obj,mur_obj,epsr_obj,k_0,dipoles,M_timepoints,off)
        # fib.make_sensors()
        # # fib.limited_aperture_sensors(NA)
        # fib.data_acquisition()
        # fib.find_resolution_limit()
        # print(fib.resolution_limit)
        # fib.save_info(dir,off/wl)

        # fib.P_estimation(101,FoV)
        # fig, ax = plt.subplots()
        # x = np.round(np.linspace(-FoV/2,FoV/2,6),5)
        # xx = np.linspace(0,100,6)
        # print(x)
        # plt.imshow(np.abs(fib.P))
        # ax.set_xticks(xx)
        # ax.set_xticklabels(x)
        # ax.set_yticks(xx)
        # ax.set_yticklabels(x)
        # plt.xlabel('x-position [wl]')
        # plt.ylabel('y-position [wl]')
        # plt.colorbar()
        # plt.savefig('images/plots/MUSIC_reconstruction_log_wl420.png',dpi=300,format='png')
        # plt.show()
