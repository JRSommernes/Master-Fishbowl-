import numpy as np
from numba import njit
from time import time
from misc_functions import *

#Do not use (Very buggy, not usable)
# def dyadic_green_FoV(sensors,xx,yy,zz,N_sensors,grid_size,k_0,presition='Single'):
#     if presition == 'Double':
#         I = np.identity(3)
#
#     if presition == 'Single':
#         sensors = sensors.astype(np.float32)
#         xx,yy,zz = xx.astype(np.float32),yy.astype(np.float32),zz.astype(np.float32)
#         k_0 = k_0.astype(np.float32)
#         # k_0 = np.float32(k_0)
#         I = np.identity(3).astype(np.float32)
#
#     r_x = sensors[0].reshape(N_sensors,1,1,1)-xx.reshape((1,grid_size,grid_size,grid_size))
#     r_y = sensors[1].reshape(N_sensors,1,1,1)-yy.reshape((1,grid_size,grid_size,grid_size))
#     r_z = sensors[2].reshape(N_sensors,1,1,1)-zz.reshape((1,grid_size,grid_size,grid_size))
#     r_p = np.array([r_x,r_y,r_z])
#
#     R = np.sqrt(np.sum((r_p)**2,axis=0))
#     R_hat = ((r_p)/R)
#     RR_hat = np.einsum('iklmn,jklmn->ijklmn',R_hat,R_hat)
#
#     # A = R_hat.transpose((1,2,3,4,0))
#     # RR_hat_1 = high_outer(A,A).transpose((4,5,0,1,2,3))
#
#     g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
#     expr_1 = np.zeros((3,3,1,1,1,1),dtype = sensors.dtype) + ((3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R)
#     expr_2 = np.zeros((3,3,1,1,1,1),dtype = sensors.dtype) + ((1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R)
#     I = (np.zeros((N_sensors,grid_size,grid_size,grid_size,3,3),dtype = sensors.dtype) + I).transpose(4,5,0,1,2,3)
#
#     G = (expr_1*RR_hat + expr_2*I).transpose((2,0,1,3,4,5))
#
#     return G

def dyadic_green(sensors,dipole_pos,N_sensors,k_0):
    r_p = sensors-dipole_pos.reshape(3,1)

    R = np.sqrt(np.sum((r_p)**2,axis=0))
    R_hat = ((r_p)/R)

    RR_hat = np.einsum('ik,jk->ijk',R_hat,R_hat)

    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
    expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
    expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R

    I = np.identity(3)
    G = (expr_1*RR_hat + expr_2*I.reshape(3,3,1)).T

    return G

#Same speed when njit
def noise_space(E_field):
    S = np.conjugate(E_field@np.conjugate(E_field).T)
    M = S.shape[0]

    eigvals,eigvecs = np.linalg.eig(S)

    dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)

    noice_idx = np.where(dist<1)[0]
    N = len(noice_idx)
    D = len(E_field)-N

    E_N = eigvecs[:,noice_idx]

    return E_N


# @njit(parallel=True)
def P_estimation(E_field,N_sensors,N_recon,sensor_radius,FoV,k_0,wl):
    E_N = noise_space(E_field)

    sensors = make_sensors(N_sensors,sensor_radius)

    dipole_pos = np.array(((0,0.5*wl,2*wl)))
    t0 = time()
    for i in range(50**3):
        dyadic_green(sensors,dipole_pos,N_sensors,k_0)
    print(time()-t0)
    exit()

    x = np.linspace(FoV[0,0],FoV[0,1],N_recon)
    y = np.linspace(FoV[1,0],FoV[1,1],N_recon)
    z = np.linspace(FoV[2,0],FoV[2,1],N_recon)

    P = np.zeros((N_recon,N_recon,N_recon),dtype=np.complex128)

    for i in range(N_recon):
        print(i)
        for j in range(N_recon):
            for k in range(N_recon):
                dipole_pos = np.array(((x[j],y[i],z[k])))
                A = dyadic_green(sensors,dipole_pos,N_sensors,k_0).reshape((3*N_sensors,3))
                P[i,j,k] = 1/(np.conjugate(A[:,0].T)@np.conjugate(E_N)@E_N.T@A[:,0])
                # P[i,j,k] += 1/(np.conjugate(A[:,1].T)@np.conjugate(E_N)@E_N.T@A[:,1])
                # P[i,j,k] += 1/(np.conjugate(A[:,2].T)@np.conjugate(E_N)@E_N.T@A[:,2])


    return P
