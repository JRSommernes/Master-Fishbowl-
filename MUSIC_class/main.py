import numpy as np
from microscope import Microscope
from fishbowl import Fishbowl
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, mu_0, c
from time import time
import json, os
from PIL import Image
from misc_functions import *

# if __name__ == '__main__':
#     dir = 'C:/Users/jso085/github/Master-Fishbowl-/MUSIC_class/images/microscope/1616588051/'
#     Im = np.zeros((11,11,11))
#     for i in range(11):
#         name = '{}.tiff'.format(i)
#         im = Image.open(dir+name)
#         Im[:,:,i] = np.array(im)
#
#     min = Im.min()
#     max = Im.max()
#     for i in range(11):
#         plt.imshow(Im[i],vmin=min,vmax=max)
#         plt.title('z = {}'.format(i))
#         plt.show()


if __name__ == '__main__':
    Mag = 60
    N_sensors = 5**2
    M_timepoints = 100
    N_recon = 101

    wl = 690e-9
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

    f_cam = 16e-2
    f_obj = f_cam/(Mag*n_cam/n_obj)
    f = [f_obj,f_cam]

    z_Interface_sub = -30e-9

    FoV = 10
    NA = 1.2
    camera_size = 6
    voxel_size = 60e-9/wl

    # fib = Fishbowl(N_sensors,f_cam,wl,n_obj,mur_obj,epsr_obj,k_0,dipoles,M_timepoints)
    # # fib.make_sensors()
    # fib.limited_aperture_sensors(NA)
    # fib.data_acquisition()
    # fib.find_resolution_limit()
    # print(fib.resolution_limit)
    #
    # mic = Microscope(Mag,N_sensors,wl,n,mur,epsr,k_0,f,NA,z_Interface_sub,dipoles,voxel_size,M_timepoints)
    # mic.create_image_stack()
    # mic.find_resolution_limit()
    # print(mic.resolution_limit)
    # exit()
    sensor_arr = np.arange(3,32,2)**2
    dir = 'single_variable/microscope/N_sensors'
    # for el in os.listdir(dir):
    #     tmp = el.split('_')[0]
    #     sensor_arr.append(int(tmp))

    counter = 0
    # N_sensors = 15**2
    M_timepoints = 100
    f_cam = 5e-2
    NA = 1.2
    for N_sensors in sensor_arr:
        M_timepoints = 100
        # N_sensors = 100
        NA = 1.2
        f_cam = 5e-2
        # f_cam = round(f_cam,5)
        f = [f_obj,f_cam]

        dipoles = np.array([[-1e-3*wl,0*wl,0*wl],[1e-3*wl,0*wl,0*wl]])



        if os.path.isfile(dir+'/{}_data_microscope.json'.format(N_sensors)):
            continue
        print(N_sensors)

        # fib = Fishbowl(N_sensors,f_cam,wl,n_obj,mur_obj,epsr_obj,k_0,dipoles,M_timepoints)
        # fib.make_sensors()
        # # fib.limited_aperture_sensors(NA)
        # # fib.plot_aperture_field()
        # fib.data_acquisition()
        # fib.find_resolution_limit()
        # print(fib.resolution_limit)
        # fib.save_info(dir,N_sensors)

        mic = Microscope(Mag,N_sensors,wl,n,mur,epsr,k_0,f,NA,z_Interface_sub,dipoles,voxel_size,M_timepoints)
        mic.create_image_stack()
        mic.find_resolution_limit()
        print(mic.resolution_limit)
        mic.save_info(dir,N_sensors)


    # for N_sensors in np.array([25,100,225]):
    #     for M_timepoints in np.array([100]):
    #         for f_cam in np.array([5e-2]):
    #             for NA in np.array([1.2]):
    #                 for voxel_size in np.array([1e-4,1e-2,1])*wl:
    #                     f_obj = f_cam/(Mag*n_cam/n_obj)
    #                     f = [f_obj,f_cam]
    #                     dipoles = np.array([[-5*wl,0*wl,0*wl],[5*wl,0*wl,0*wl]])
    #
    #                     dir = 'increasing_voxel_size'
    #
    #                     if os.path.isfile(dir+'/{}_data_fishbowl.json'.format(counter)):
    #                         counter+=1
    #                         continue
    #                     print(counter)
    #
    #                     fib = Fishbowl(N_sensors,f_cam,wl,n_obj,mur_obj,epsr_obj,k_0,dipoles,M_timepoints,voxel_size)
    #                     # fib.make_sensors()
    #                     fib.limited_aperture_sensors(NA)
    #                     # fib.plot_aperture_field()
    #                     fib.data_acquisition()
    #                     fib.find_resolution_limit()
    #                     print(fib.resolution_limit)
    #                     fib.save_info(dir,counter)


    # N_sensors = 225
    # M_timepoints = 100
    # f_cam = 0.2
    # NA = 1.2
    #
    # dipoles = np.array([[-1.2e-05,0.0,0.0],[1.2e-05,0.0,0.0]])
    # mic = Microscope(Mag,N_sensors,wl,n,mur,epsr,k_0,f,NA,z_Interface_sub,dipoles,voxel_size,M_timepoints)
    # mic.create_image_stack()
    # mic.find_resolution_limit()
    # print(mic.resolution_limit)
    # mic.save_info(dir,counter)

        # counter+=1
