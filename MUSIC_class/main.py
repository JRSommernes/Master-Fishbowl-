import numpy as np
from microscope import Microscope
from fishbowl import Fishbowl
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, mu_0, c
from time import time
import json, os
from PIL import Image

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


# 60-120 nm

if __name__ == '__main__':
    Mag = 60
    N_sensors = 5**2
    M_timepoints = 50
    N_recon = 11

    wl = 690e-9
    freq = c/wl

    n_obj = 1.33
    n_sub = 4.3
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

    NA = 1.2
    z_Interface_sub = -30e-9

    FoV = 1
    camera_size = 6

    dipoles = np.array([[-0.8*wl,0*wl,0*wl],[0.8*wl,0*wl,0*wl]])

    n = [n_obj,n_sub]
    mur = [mur_obj,mur_sub]
    epsr = [epsr_obj,epsr_sub]
    fib = Fishbowl(N_sensors,f_cam,wl,n,mur,epsr,k_0,NA,z_Interface_sub,dipoles,M_timepoints)
    # fib.make_bowl_sensors()
    fib.limited_aperture_sensors(NA)
    fib.data_aquisition()
    # fib.find_resolution_limit()
    fib.reconstruct_image(N_recon)
    plt.imshow(np.abs(fib.P))
    plt.show()

    """
    Something funky here. Contrast not as high as in old MUSIC algorithm.
    """





    # mic = Microscope(Mag,N_sensors,wl,n,mur,epsr,k_0,f,NA,z_Interface_sub,dipoles,camera_size,M_timepoints)
    # mic.create_image_stack()
    # mic.find_resolution_limit()
    #
    # diff = mic.resolution_limit
    # FoV = diff*1.5
    # mic.reconstruct_image(FoV,N_recon)

    # plt.imshow(np.abs(mic.P))
    # plt.show()
    #
    # dir = 'C:/Users/jso085/github/Master-Fishbowl-/MUSIC_class/images/microscope'
    # mic.save_info(dir)
