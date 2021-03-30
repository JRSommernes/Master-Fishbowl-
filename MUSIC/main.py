import numpy as np
try:
    import cupy as cp
except:
    pass
from misc_functions import *
from imaging import *
from MUSIC import *
from microscope_data import *
import json
import os

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

    # sensor_radius = 100*wl
    sensor_radius = 16e-2

    # dipoles = np.array([[1*wl, 0*wl, 0*wl]])
    # dipoles = np.array([[0.4*wl,0.4*wl,0.4*wl],[-0.4*wl,0.4*wl,0.4*wl]])
    dipoles = np.array([[-0.03*wl,0*wl,0*wl],[0.03*wl,0*wl,0*wl]])
    # dipoles = np.array([[0*wl,0.05*wl,0*wl],[0*wl,-0.05*wl,0*wl]])


    FoV = np.array([[-0.2*wl,0.2*wl],
                    [-0.2*wl,0.2*wl],
                    [-0.2*wl,0.2*wl]])
    # FoV = np.array([[-1.5*wl,1.5*wl],
    #                 [-1.5*wl,1.5*wl],
    #                 [-1.5*wl,1.5*wl]])

    N_sensors = 25
    N_emitters = 100
    M_inputs = 100
    N_recon = 101

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

    E_sensors,sensors = data_acquisition(dipoles,wl,M_inputs,sensor_radius,N_sensors,k_0)
    P = P_estimation(E_sensors,sensors,N_recon,FoV,k_0,target='cpu')
    # plot_sensor_field(sensors,E_sensors)


    # Mag = 60
    # N_sensors = 61**2
    # microscope_greens(dipoles,wl,M_inputs,N_sensors,k_0,Mag)

    plt.imshow(np.abs(P[:,:,0]))
    plt.show()

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
