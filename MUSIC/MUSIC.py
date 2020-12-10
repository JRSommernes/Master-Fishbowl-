import numpy as np
# import gnumpy as gnp
from numba import njit, guvectorize, float64, complex128, int64
from time import time
from misc_functions import dyadic_green, high_inner, loadbar
from multiprocessing import Pool
from functools import partial

# # @njit
# def dyadic_green_FoV(sensors,xx,yy,zz,N_sensors,grid_size,k_0,presition='Double'):
#     if presition == 'Double':
#         I = np.identity(3)
#
#     if presition == 'Single':
#         sensors = sensors.astype(np.float32)
#         xx,yy,zz = xx.astype(np.float32),yy.astype(np.float32),zz.astype(np.float32)
#         k_0 = k_0.astype(np.float32)
#         I = np.identity(3).astype(np.float32)
#
#     r_x = sensors[0].reshape(N_sensors,1,1,1)-xx.reshape((1,grid_size,grid_size,grid_size))
#     r_y = sensors[1].reshape(N_sensors,1,1,1)-yy.reshape((1,grid_size,grid_size,grid_size))
#     r_z = sensors[2].reshape(N_sensors,1,1,1)-zz.reshape((1,grid_size,grid_size,grid_size))
#     r_p = np.array((r_x,r_y,r_z))
#
#     R = np.sqrt(np.sum((r_p)**2,axis=0))
#     R_hat = ((r_p)/R)
#     RR_hat = np.einsum('iklmn,jklmn->ijklmn',R_hat,R_hat)
#
#     g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
#     expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
#     expr_1 = np.broadcast_to(expr_1,RR_hat.shape)
#
#     expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R
#     expr_2 = np.broadcast_to(expr_2,RR_hat.shape)
#
#     I = np.broadcast_to(I,(N_sensors,grid_size,grid_size,grid_size,3,3))
#     I = I.transpose(4,5,0,1,2,3)
#
#     G = (expr_1*RR_hat + expr_2*I).transpose((2,0,1,3,4,5))
#
#     return G

# @njit
def dyadic_green_FoV_2D(sensors,r_p,N_sensors,grid_size,k_0):
    R = np.sqrt(np.sum((r_p)**2,axis=0))
    R_hat = ((r_p)/R)
    RR_hat = np.einsum('iklm,jklm->ijklm',R_hat,R_hat)

    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
    expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
    expr_1 = np.broadcast_to(expr_1,RR_hat.shape)

    expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R
    expr_2 = np.broadcast_to(expr_2,RR_hat.shape)

    I = np.identity(3)
    I = np.broadcast_to(I,(N_sensors,grid_size,grid_size,3,3))
    I = I.transpose(3,4,0,1,2)

    G = (expr_1*RR_hat + expr_2*I).transpose((2,0,1,3,4))

    return G


#Same speed when njit
def noise_space(E_field):
    S = E_field@np.conjugate(E_field).T

    eigvals,eigvecs = np.linalg.eig(S)

    dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)

    noice_idx = np.where(dist<1)[0]
    N = len(noice_idx)
    D = len(E_field)-N

    E_N = eigvecs[:,noice_idx]

    return np.ascontiguousarray(E_N)

# @njit(parallel = True)
def P_calc_2D(A_fov,E_N):
    a,b,c,d = A_fov.shape

    A = A_fov.reshape(-1, A_fov.shape[-1])
    B =  E_N.reshape(-1, E_N.shape[-1])

    P_fov_1 = np.conjugate(A)@B
    P_fov_2 = A@np.conjugate(B)

    P_fov_1 = P_fov_1.reshape(a,b,c,P_fov_1.shape[-1])
    P_fov_2 = P_fov_2.reshape(a,b,c,P_fov_2.shape[-1])

    return P_fov_1, P_fov_2

# @njit(parallel=True)
def P_estimation(E_field,sensors,N_recon,FoV,k_0):
    N_sensors = sensors.shape[1]

    E_N = noise_space(E_field)

    x = np.linspace(FoV[0,0],FoV[0,1],N_recon)
    y = np.linspace(FoV[1,0],FoV[1,1],N_recon)
    z = np.linspace(FoV[2,0],FoV[2,1],N_recon)

    xx,yy = np.meshgrid(x,y)
    shape_1 = np.append(N_sensors,np.ones(len(xx.shape),dtype=int))
    shape_2 = np.append(1,xx.shape)
    P_t = np.zeros((N_recon,N_recon,N_recon),dtype=np.complex128)
    for i,zz in enumerate(z):
        loadbar(i,len(z))
        r_x = sensors[0].reshape(shape_1)-xx.reshape(shape_2)
        r_y = sensors[1].reshape(shape_1)-yy.reshape(shape_2)
        r_z = sensors[2].reshape(shape_1)-zz*np.ones(shape_2)
        r_p = np.array((r_x,r_y,r_z))

        # t0 = time()
        A_fov_plane = dyadic_green_FoV_2D(sensors,r_p,N_sensors,N_recon,k_0)
        A_fov_plane = np.ascontiguousarray((A_fov_plane.reshape(3*N_sensors,3,N_recon,N_recon)).T)
        # print(time()-t0)

        # t0 = time()
        P_fov_plane_1, P_fov_plane_2 = P_calc_2D(A_fov_plane,E_N)
        # print(time()-t0)
        P_fov_plane = (1/np.einsum('ijkl,ijkl->ijk',P_fov_plane_1,P_fov_plane_2)).T
        P_fov_plane = np.sum(P_fov_plane,axis=0)
        P_t[:,:,i] = P_fov_plane
        # if i == 3:
        #     exit()

    return P_t
