import numpy as np
from microscope import Microscope
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

    dipoles = np.array([[0.4*wl,0*wl,0*wl],[0*wl,0*wl,0*wl]])

    mic = Microscope(Mag,N_sensors,wl,n,mur,epsr,k_0,f,NA,z_Interface_sub,dipoles)


    mic.create_image_stack(camera_size,M_timepoints)
    mic.reconstruct_image(FoV,camera_size,N_recon,dipoles[0,2])

    # dir = 'C:/Users/jso085/github/Master-Fishbowl-/MUSIC_class/images/microscope'
    # mic.save_image_stack(dir)
    plt.imshow(np.abs(mic.P))
    plt.show()
