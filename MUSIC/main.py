import numpy as np
try:
    import cupy as cp
except:
    pass
from misc_functions import *
from imaging import *
from MUSIC import *
from fishbowl import *
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

    sensor_radius = 10*wl
    # dipoles = np.array([[0*wl,0*wl,0*wl]])

    FoV = np.array([[-2*wl,2*wl],
                    [-2*wl,2*wl],
                    [-2*wl,2*wl]])

    N_sensors = 50
    M_inputs = 100
    N_recon = 101

    dipoles = np.array([[0.8*wl,0*wl,0*wl],
                        [-0.8*wl,0*wl,0*wl],
                        [0*wl,0.8*wl,0*wl],
                        [0.8*wl,0*wl,0.8*wl],
                        [-0.8*wl,0*wl,0.8*wl],
                        [0*wl,0.8*wl,0.8*wl],
                        [0.8*wl,0*wl,-0.8*wl],
                        [-0.8*wl,0*wl,-0.8*wl],
                        [0*wl,0.8*wl,-0.8*wl]])

    # dipoles = np.random.uniform(-1.5*wl,1.5*wl,(9,3))

    E_sensors,sensors = data_acquisition(dipoles,wl,M_inputs,sensor_radius,N_sensors,k_0)

    # plot_sensor_field(sensors,E_sensors)

    # P = P_estimation(E_sensors,sensors,N_recon,FoV,k_0,target='cpu')

    pos_list = np.array([[0,0,0],[0,0.8*wl,0],[wl,0,wl],[0.8*wl,0,0]])

    for pos in pos_list:
        T1, T2 = fishbowl(E_sensors,sensors,wl,k_0,pos)
        # test_1 = np.sum(T[:3],axis=0)
        # test_2 = np.sum(T[3:6],axis=0)
        # plt.plot(np.abs(test_1),c='b')
        # plt.plot(np.abs(test_2),c='r')
        for i in range(3):
            for j in range(3):
                plt.plot(np.abs(T1[i,j]))
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
