import numpy as np
from microscope import Microscope
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, mu_0, c


if __name__ == '__main__':
    Mag = 60
    N_sensors = 61**2

    wl = 561e-9
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

    dipoles = np.array([[0.4*wl,0.4*wl,0.4*wl],[-0.4*wl,0.4*wl,0.4*wl]])

    mic = Microscope(Mag,N_sensors,wl,n,mur,epsr,k_0,f,NA,z_Interface_sub,dipoles)

    z_cam = 0
    FoV = 6
    mic.image_field(z_cam,FoV)
