import numpy as np
import matplotlib.pyplot as plt
from time import time
from MUSIC import *
from misc_functions import *
from imaging import *

def FoV_reconstruction(E_sensors,N_sensors,N_recon,sensor_radius,FoV,k_0,wl):
    E_N = noise_space(E_sensors)
    P = P_estimation(E_sensors,N_sensors,N_recon,sensor_radius,FoV,k_0,wl,E_N)
    return P

if __name__ == '__main__':
    eps_0 = 8.8541878128e-12
    mu_0 = 4*np.pi*10**-7
    c_0 = 1/np.sqrt(eps_0*mu_0)

    wl = 690e-9
    freq = c_0/wl
    k_0 = 2*np.pi*freq*np.sqrt(eps_0*mu_0)
    omega = 2*np.pi*freq

    sensor_radius = 10*wl
    dipoles = np.array([[0.7*wl,0,0]])
    # dipoles = np.array([[0,0,0]])
    # FoV = 4*wl
    FoV = np.array([[-2*wl,2*wl],[-2*wl,2*wl],[-2*wl,2*wl]])

    N_sensors = 50
    M_inputs = 100
    N_recon = 51

    E_sensors = data_acquisition(dipoles,wl,M_inputs,sensor_radius,N_sensors,k_0)
    P = FoV_reconstruction(E_sensors,N_sensors,N_recon,sensor_radius,FoV,k_0,wl)

    save_stack(P,'C:/python/Master (Fishbowl)/MUSIC/p_theta_test')
