import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from time import time
from misc_functions import *
from imaging import *

# @njit
def dyadic_green_FoV(sensors,xx,yy,zz,N_sensors,grid_size,k_0,presition='Single'):
    if presition == 'Double':
        I = np.identity(3)

    if presition == 'Single':
        sensors = sensors.astype(np.float32)
        xx,yy,zz = xx.astype(np.float32),yy.astype(np.float32),zz.astype(np.float32)
        k_0 = k_0.astype(np.float32)
        # k_0 = np.float32(k_0)
        I = np.identity(3).astype(np.float32)

    r_x = sensors[0].reshape(N_sensors,1,1,1)-xx.reshape((1,grid_size,grid_size,grid_size))
    r_y = sensors[1].reshape(N_sensors,1,1,1)-yy.reshape((1,grid_size,grid_size,grid_size))
    r_z = sensors[2].reshape(N_sensors,1,1,1)-zz.reshape((1,grid_size,grid_size,grid_size))
    r_p = np.array([r_x,r_y,r_z])

    R = np.sqrt(np.sum((r_p)**2,axis=0))
    R_hat = ((r_p)/R)
    RR_hat = np.einsum('iklmn,jklmn->ijklmn',R_hat,R_hat)

    print(R_hat[:,0,27,25,25])
    #Ikke riktig til hit
    exit()

    # A = R_hat.transpose((1,2,3,4,0))
    # RR_hat_1 = high_outer(A,A).transpose((4,5,0,1,2,3))

    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
    expr_1 = np.zeros((3,3,1,1,1,1),dtype = sensors.dtype) + ((3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R)
    expr_2 = np.zeros((3,3,1,1,1,1),dtype = sensors.dtype) + ((1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R)
    I = (np.zeros((N_sensors,grid_size,grid_size,grid_size,3,3),dtype = sensors.dtype) + I).transpose(4,5,0,1,2,3)

    G = (expr_1*RR_hat + expr_2*I)

    return G

def noise_space(E_field):
    S = np.conjugate(E_field@np.conjugate(E_field).T)
    M = S.shape[0]

    eigvals,eigvecs = np.linalg.eig(S)
    mat = np.concatenate((eigvals.real.reshape(M,1),eigvals.imag.reshape(M,1)),axis=1)

    min = np.min(eigvals)
    min_idx = np.where(eigvals==min)[0][0]
    dist = cdist(mat,mat)[min_idx]


    noice_idx = np.where(dist<1.1)[0]
    N = len(noice_idx)
    D = len(E_field)-N

    #Dunno if :,noice_idx is right, or should be noice_idx
    E_N = eigvecs[:,noice_idx]

    return E_N#@np.conjugate(E_N.T)


# @njit(parallel=True)
def P_estimation(E_field,N_sensors,N_recon,sensor_radius,FoV,k_0,wl,E_N):
    sensors = make_sensors(N_sensors,sensor_radius)

    x = np.linspace(FoV[0,0],FoV[0,1],N_recon)
    y = np.linspace(FoV[1,0],FoV[1,1],N_recon)
    z = np.linspace(FoV[2,0],FoV[2,1],N_recon)

    xx,yy,zz = np.meshgrid(x,y,z)
    G_FoV = dyadic_green_FoV(sensors,xx,yy,zz,N_sensors,N_recon,k_0,presition='Single')
    A_FoV = G_FoV.reshape((3*N_sensors,3,N_recon,N_recon,N_recon))

    #Detta blir riktig
    dipole_pos = np.array([[0.7*wl,0,0]])
    A_test = dyadic_green(sensors,dipole_pos,N_sensors,k_0).reshape((3*N_sensors,3))
    print(np.conjugate(A_test[:,0].T)@np.conjugate(E_N)@E_N.T@A_test[:,0])
    #Formelen e rett, men A_FoV e feil


    exit()

    # A_0 = np.conjugate(A_FoV[:,0]).transpose((1,2,3,0))@np.conjugate(E_N)
    # B_0 = (A_FoV[:,0].transpose((1,2,3,0)))@E_N
    # P_0 = 1/high_inner(A_0,B_0)
    #
    #
    # A_1 = np.conjugate(A_FoV[:,1]).transpose((1,2,3,0))@E_N
    # B_1 = np.copy(A_FoV[:,1].transpose((1,2,3,0)),order='C')
    # P_1 = 1/high_inner(A_1,B_1)
    #
    # A_2 = np.conjugate(A_FoV[:,2]).transpose((1,2,3,0))@E_N
    # B_2 = np.copy(A_FoV[:,2].transpose((1,2,3,0)),order='C')
    # P_2 = 1/high_inner(A_2,B_2)
    #
    # P = P_0

    return P
