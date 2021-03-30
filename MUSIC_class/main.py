import numpy as np
from microscope import Microscope
from new_fishbowl import Fishbowl
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

    FoV = 1
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

    counter = 288
    N_sensors = 15**2
    # for N_sensors in np.array([5,10,15])**2:
    for M_timepoints in np.array([10,40,70,100]):
        for f_cam in np.array([5e-2,1e-1,2e-1]):
            for NA in np.array([0.8,1,1.2]):
                f_obj = f_cam/(Mag*n_cam/n_obj)
                f = [f_obj,f_cam]
                dipoles = np.array([[-1*wl,0*wl,0*wl],[1*wl,0*wl,0*wl]])

                dir = 'C:/Users/jso085/github/Master-Fishbowl-/MUSIC_class/resolutions_2'

                fib = Fishbowl(N_sensors,f_cam,wl,n_obj,mur_obj,epsr_obj,k_0,dipoles,M_timepoints)
                # fib.make_sensors()
                fib.limited_aperture_sensors(NA)
                fib.data_acquisition()
                fib.find_resolution_limit()
                print(fib.resolution_limit)
                fib.save_info(dir,counter)

                for voxel_size in np.array([60,80,100,120])*1e-9/wl:

                    mic = Microscope(Mag,N_sensors,wl,n,mur,epsr,k_0,f,NA,z_Interface_sub,dipoles,voxel_size,M_timepoints)
                    mic.create_image_stack()
                    mic.find_resolution_limit()
                    print(mic.resolution_limit)
                    mic.save_info(dir,counter)

                    counter+=1
