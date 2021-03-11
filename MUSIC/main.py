import numpy as np
try:
    import cupy as cp
except:
    pass
from misc_functions import *
from imaging import *
from MUSIC import *
# from fishbowl import *
import json
import os
from intensity_only_MUSIC import *
from scattering_imaging import *
from intensity_scattering_MUSIC import *
from vectorial_scattering import *

if __name__ == '__main__':
    eps_0 = 8.8541878128e-12
    mu_0 = 4*np.pi*10**-7
    c_0 = 1/np.sqrt(eps_0*mu_0)

    wl = 690e-9
    freq = c_0/wl
    k_0 = 2*np.pi*freq*np.sqrt(eps_0*mu_0)
    omega = 2*np.pi*freq

    n0 = 1
    n1 = 1.33

    sensor_radius = 100*wl
    dipoles = np.array([[0*wl, 1*wl, 0*wl]])
    dipoles = np.array([[1*wl,-1*wl,0*wl],[-1*wl,1*wl,0*wl]])

    FoV = np.array([[-2*wl,2*wl],
                    [-2*wl,2*wl],
                    [-2*wl,2*wl]])

    N_sensors = 50
    N_emitters = 100
    M_inputs = 100
    N_recon = 49

    # dipoles = np.array([[0.8*wl,0*wl,0*wl],
    #                     [-0.8*wl,0*wl,0*wl],
    #                     [0*wl,0.8*wl,0*wl],
    #                     [0.8*wl,0*wl,0.8*wl],
    #                     [-0.8*wl,0*wl,0.8*wl],
    #                     [0*wl,0.8*wl,0.8*wl],
    #                     [0.8*wl,0*wl,-0.8*wl],
    #                     [-0.8*wl,0*wl,-0.8*wl],
    #                     [0*wl,0.8*wl,-0.8*wl]])

    # dipoles = np.random.uniform(-1.5*wl,1.5*wl,(5,3))

    # E_sensors,sensors = data_acquisition(dipoles,wl,M_inputs,sensor_radius,N_sensors,k_0)

    # plot_sensor_field(sensors,E_sensors)

    # P = P_estimation(E_sensors,sensors,N_recon,FoV,k_0,target='cpu')

    # I = np.abs(E_sensors)**2

    # intensity_P_estimation(I,sensors,N_recon,FoV,k_0,E_sensors)

    #Working scattering MUSIC
    E_scattering,E_incident,sensors,emitters = scattering_data(dipoles,sensor_radius,N_sensors,N_emitters,k_0,n1,n0)
    I = np.abs(E_scattering+E_incident)**2
    scatter_MUSIC(I,sensors,emitters,N_recon,FoV,k_0,E_scattering,E_incident,dipoles)


    # E_sensors,sensors,emitters = vectorial_scattering_data(dipoles,sensor_radius,N_sensors,N_emitters,k_0,n1,n0)
    # I = np.abs(E_sensors)**2
    # scatter_MUSIC(I,sensors,emitters,N_recon,FoV,k_0,E_sensors,dipoles)
    exit()

    current = '9_dipoles'
    dir = 'images'

    data = {'Num_dipoles' : len(dipoles),
            'N_recon' : N_recon,
            'N_sensors' : N_sensors,
            'M_orientations' : M_inputs,
            'FoV' : FoV.tolist(),
            'Dipole_positions' : dipoles.tolist()}

    with open(dir+'/'+current+'/'+"test.json", 'w') as output:
        json.dump(data, output, indent=4)

    save_stack(P,dir+'/'+current+'/'+'image')
