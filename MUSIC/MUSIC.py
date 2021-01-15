import numpy as np
try:
    import cupy as cp
except:
    pass
from time import time
from misc_functions import dyadic_green, high_inner, loadbar
from multiprocessing import Pool
from functools import partial

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

# @njit
def dyadic_green_FoV_2D(sensors,xx,yy,zz,N_sensors,grid_size,k_0):
    shape_1 = np.append(N_sensors,np.ones(len(xx.shape),dtype=int))
    shape_2 = np.append(1,xx.shape)

    r_x = sensors[0].reshape(shape_1)-xx.reshape(shape_2)
    r_y = sensors[1].reshape(shape_1)-yy.reshape(shape_2)
    r_z = sensors[2].reshape(shape_1)-zz*np.ones(shape_2)
    r_p = np.array((r_x,r_y,r_z))

    R = np.sqrt(np.sum((r_p)**2,axis=0))
    R_hat = ((r_p)/R)
    RR_hat = np.einsum('iklm,jklm->ijklm',R_hat,R_hat)

    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
    expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
    expr_1 = np.broadcast_to(expr_1,RR_hat.shape)

    expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R
    expr_2 = np.broadcast_to(expr_2,RR_hat.shape)

    I = np.broadcast_to(I,(N_sensors,grid_size,grid_size,3,3))
    I = I.transpose(3,4,0,1,2)

    G = (expr_1*RR_hat + expr_2*I).transpose((2,0,1,3,4))

    return G

# # @njit
# def dyadic_green_FoV_2D(sensors,r_p,N_sensors,grid_size,k_0):
#     R = np.sqrt(np.sum((r_p)**2,axis=0))
#     R_hat = ((r_p)/R)
#     RR_hat = np.einsum('iklm,jklm->ijklm',R_hat,R_hat)
#
#     g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
#     expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
#     expr_1 = np.broadcast_to(expr_1,RR_hat.shape)
#
#     expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R
#     expr_2 = np.broadcast_to(expr_2,RR_hat.shape)
#
#     I = np.identity(3)
#     I = np.broadcast_to(I,(N_sensors,grid_size,grid_size,3,3))
#     I = I.transpose(3,4,0,1,2)
#
#     G = (expr_1*RR_hat + expr_2*I).transpose((2,0,1,3,4))
#
#     return G

def dyadic_green_FoV_2D_cuda(sensors,xx,yy,zz,N_sensors,grid_size,k_0):
    sensors = cp.array(sensors)
    xx,yy = cp.array(xx),cp.array(yy)
    I = cp.identity(3)

    shape_1 = tuple([N_sensors]+[1]*len(xx.shape))
    shape_2 = tuple([1]+list(xx.shape))

    r_x = sensors[0].reshape(shape_1)-xx.reshape(shape_2)
    r_y = sensors[1].reshape(shape_1)-yy.reshape(shape_2)
    r_z = sensors[2].reshape(shape_1)-zz*cp.ones(shape_2)
    r_p = cp.array((r_x,r_y,r_z))

    R = cp.sqrt(cp.sum((r_p)**2,axis=0))
    R_hat = ((r_p)/R)
    RR_hat = cp.einsum('iklm,jklm->ijklm',R_hat,R_hat)

    g_R = cp.exp(1j*k_0*R)/(4*np.pi*R)
    expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
    expr_1 = cp.broadcast_to(expr_1,RR_hat.shape)

    expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R
    expr_2 = cp.broadcast_to(expr_2,RR_hat.shape)

    I = cp.broadcast_to(I,(N_sensors,grid_size,grid_size,3,3))
    I = I.transpose(3,4,0,1,2)

    G = (expr_1*RR_hat + expr_2*I).transpose((2,0,1,3,4))

    return G

# @njit(parallel = True)
def P_calc_2D(A_fov,E_N):
    a,b,c,d = A_fov.shape

    A = A_fov.reshape(-1, A_fov.shape[-1])
    B =  E_N.reshape(-1, E_N.shape[-1])

    P_fov_1 = np.conjugate(A)@B
    P_fov_2 = A@np.conjugate(B)

    P_fov_1 = P_fov_1.reshape(a,b,c,P_fov_1.shape[-1])
    P_fov_2 = P_fov_2.reshape(a,b,c,P_fov_2.shape[-1])

    P_fov_plane = (1/np.einsum('ijkl,ijkl->ijk',P_fov_1,P_fov_2)).T
    P_fov_plane = np.sum(P_fov_plane,axis=0)

    return P_fov_plane

def P_calc_2D_cuda(A_fov,E_N):
    a,b,c,d = A_fov.shape
    A = A_fov.reshape(-1, A_fov.shape[-1])
    B =  E_N.reshape(-1, E_N.shape[-1])

    A = cp.array(A)
    B = cp.array(B)

    P_fov_1 = cp.matmul(cp.conjugate(A),B)
    P_fov_2 = cp.matmul(A,cp.conjugate(B))

    P_fov_1 = P_fov_1.reshape(a,b,c,P_fov_1.shape[-1])
    P_fov_2 = P_fov_2.reshape(a,b,c,P_fov_2.shape[-1])

    P_fov_plane_cuda = (1/cp.einsum('ijkl,ijkl->ijk',P_fov_1,P_fov_2)).T
    P_fov_plane_cuda = np.sum(P_fov_plane_cuda,axis=0)

    return P_fov_plane_cuda

# @njit(parallel=True)
def P_estimation(E_field,sensors,N_recon,FoV,k_0,target='cuda'):
    N_sensors = sensors.shape[1]

    E_N = noise_space(E_field)

    x = np.linspace(FoV[0,0],FoV[0,1],N_recon)
    y = np.linspace(FoV[1,0],FoV[1,1],N_recon)
    z = np.linspace(FoV[2,0],FoV[2,1],N_recon)

    xx,yy = np.meshgrid(x,y)
    if target == 'cuda':
        P_t = cp.zeros((N_recon,N_recon,N_recon),dtype=np.complex128)
        for i,zz in enumerate(z):
            print(int(i/N_recon*100))
            #Every calculation is one kernel, when reaching 1024? kernels
            #the queue overflows and computation time goes up?
            A_fov_plane = dyadic_green_FoV_2D_cuda(sensors,xx,yy,zz,N_sensors,N_recon,k_0)
            A_fov_plane = cp.ascontiguousarray((A_fov_plane.reshape(3*N_sensors,3,N_recon,N_recon)).T)

            P_fov_plane = P_calc_2D_cuda(A_fov_plane,E_N)
            P_t[:,:,i] = P_fov_plane

        P_t = P_t.get()


    elif target == 'cpu':
        P_t = np.zeros((N_recon,N_recon,N_recon),dtype=np.complex128)
        for i,zz in enumerate(z):
            print(int(i/N_recon*100))
            A_fov_plane = dyadic_green_FoV_2D(sensors,xx,yy,zz,N_sensors,N_recon,k_0)
            A_fov_plane = np.ascontiguousarray((A_fov_plane.reshape(3*N_sensors,3,N_recon,N_recon)).T)

            P_fov_plane = P_calc_2D(A_fov_plane,E_N)
            P_t[:,:,i] = P_fov_plane

    return P_t
