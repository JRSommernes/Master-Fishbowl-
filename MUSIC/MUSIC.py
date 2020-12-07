import numpy as np
from numba import njit, guvectorize, float64, complex128, int64
from time import time
from misc_functions import dyadic_green, high_inner

def dyadic_green_FoV(sensors,xx,yy,zz,N_sensors,grid_size,k_0,presition='Double'):
    if presition == 'Double':
        I = np.identity(3)

    if presition == 'Single':
        sensors = sensors.astype(np.float32)
        xx,yy,zz = xx.astype(np.float32),yy.astype(np.float32),zz.astype(np.float32)
        k_0 = k_0.astype(np.float32)
        I = np.identity(3).astype(np.float32)

    r_x = sensors[0].reshape(N_sensors,1,1,1)-xx.reshape((1,grid_size,grid_size,grid_size))
    r_y = sensors[1].reshape(N_sensors,1,1,1)-yy.reshape((1,grid_size,grid_size,grid_size))
    r_z = sensors[2].reshape(N_sensors,1,1,1)-zz.reshape((1,grid_size,grid_size,grid_size))
    r_p = np.array([r_x,r_y,r_z])

    R = np.sqrt(np.sum((r_p)**2,axis=0))
    R_hat = ((r_p)/R)
    RR_hat = np.einsum('iklmn,jklmn->ijklmn',R_hat,R_hat)

    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
    expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
    expr_1 = np.broadcast_to(expr_1,RR_hat.shape)

    expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R
    expr_2 = np.broadcast_to(expr_2,RR_hat.shape)

    I = np.broadcast_to(I,(N_sensors,grid_size,grid_size,grid_size,3,3))
    I = I.transpose(4,5,0,1,2,3)

    G = (expr_1*RR_hat + expr_2*I).transpose((2,0,1,3,4,5))

    return G

#Same speed when njit
def noise_space(E_field):
    S = E_field@np.conjugate(E_field).T

    eigvals,eigvecs = np.linalg.eig(S)

    dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)

    noice_idx = np.where(dist<1)[0]
    N = len(noice_idx)
    D = len(E_field)-N
    print(D)

    E_N = eigvecs[:,noice_idx]

    return E_N

# @njit(parallel=True)
def P_estimation(E_field,sensors,N_recon,FoV,k_0):
    N_sensors = sensors.shape[1]

    E_N = noise_space(E_field)

    x = np.linspace(FoV[0,0],FoV[0,1],N_recon)
    y = np.linspace(FoV[1,0],FoV[1,1],N_recon)
    z = np.linspace(FoV[2,0],FoV[2,1],N_recon)

    xx,yy,zz = np.meshgrid(x,y,z)
    A_fov = dyadic_green_FoV(sensors,xx,yy,zz,N_sensors,N_recon,k_0).reshape(3*N_sensors,3,N_recon,N_recon,N_recon)

    P_fov_1 = np.ascontiguousarray(np.conjugate(A_fov.T)@E_N)
    P_fov_2 = np.ascontiguousarray(A_fov.T@np.conjugate(E_N))
    P_fov = (1/np.einsum('ijklm,ijklm->ijkl',P_fov_1,P_fov_2)).T
    P_fov = np.sum(P_fov,axis=0)

    ###########################################################################################################################

    # N_sensors = sensors.shape[1]
    #
    # # E_x = E_field[:N_sensors]
    # E_x = np.zeros((50,100),dtype=np.complex128)
    # E_y = E_field[N_sensors:2*N_sensors]
    # E_z = E_field[2*N_sensors:]
    #
    # E_N_x = noise_space(E_x)
    # E_N_y = noise_space(E_y)
    # E_N_z = noise_space(E_z)
    # E_N = [E_N_x,E_N_y,E_N_z]
    # P = np.zeros((N_recon,N_recon,N_recon),dtype=np.complex128)
    # for i in range(3):
    #     for j in range(3):
    #         A = A_fov[:,i,j]
    #
    #         P_1 = np.ascontiguousarray(np.conjugate(A.T)@E_N[i])
    #         P_2 = np.ascontiguousarray(A.T@np.conjugate(E_N[i]))
    #         P += (1/high_inner(P_1,P_2)).T
    #
    # P_fov = np.zeros((N_recon,N_recon,N_recon),dtype=np.complex128)
    # for i in range(3):
    #     P_fov_1 = np.ascontiguousarray(np.conjugate(A_fov[:,i].T)@E_N)
    #     P_fov_2 = np.ascontiguousarray(A_fov[:,i].T@np.conjugate(E_N))
    #     P_fov += (1/high_inner(P_fov_1,P_fov_2)).T

    exit()
    return P_fov
