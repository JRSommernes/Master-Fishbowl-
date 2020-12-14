import numpy as np
import cupy as cp
from misc_functions import *
from imaging import *
from MUSIC import *
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
    dipoles = np.array([[0*wl,0*wl,0*wl]])

    FoV = np.array([[-2*wl,2*wl],
                    [-2*wl,2*wl],
                    [-2*wl,2*wl]])

    N_sensors = 50
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

    # dipoles = np.random.uniform(-1.5*wl,1.5*wl,(9,3))

    E_sensors,sensors = data_acquisition(dipoles,wl,M_inputs,sensor_radius,N_sensors,k_0)

    n = 50
    pos = np.array([[0,0,0]])
    tmp = np.linspace(-0.1*wl,0.1*wl,n)
    tmp_x = np.broadcast_to(tmp,(n,n)).flatten()
    tmp_y = np.broadcast_to(tmp,(n,n)).T.flatten()
    tmp_z = np.zeros_like(tmp_x)
    tmp = np.array((tmp_x,tmp_y,tmp_z))
    im = dyadic_green(tmp,pos,k_0)
    im = im.reshape(n,n,3,3)
    im = im[:,:,0]@np.array((1,0,0))
    plt.imshow(np.log(np.abs(im)))
    plt.show()
    exit()

    P = P_estimation(E_sensors,sensors,N_recon,FoV,k_0)


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
