import numpy as np
try:
    import cupy as cp
except:
    pass
from time import time
from misc_functions import dyadic_green, high_inner, loadbar
import matplotlib.pyplot as plt

#Same speed when njit
def noise_space(E_field):
    S = E_field@np.conjugate(E_field).T

    eigvals,eigvecs = np.linalg.eig(S)

    dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)

    noise_idx = np.where(np.log(dist)/np.log(dist[0])<0.5)[0]

    E_N = eigvecs[:,noise_idx]

    return np.ascontiguousarray(E_N)

# @njit
def dyadic_green_FoV_2D(sensors,xx,yy,zz,N_sensors,grid_size,k_0):
    I = np.identity(3)
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

def intensity_P_estimation(I,sensors,N_recon,FoV,k_0):
    N_sensors = sensors.shape[1]

    I_x = I[:I.shape[0]//3]

    I_N = noise_space(I_x)

    x = np.linspace(FoV[0][0],FoV[0][1],N_recon)
    y = np.linspace(FoV[1][0],FoV[2][1],N_recon)
    z = np.linspace(FoV[1][0],FoV[2][1],N_recon)
    xx,yy = np.meshgrid(x,y)

    G = dyadic_green_FoV_2D(sensors,xx,yy,z[N_recon//2],N_sensors,N_recon,k_0)[:,0]

    print(G.shape,I_N.shape)

    # test = G[:,0]
    # test = np.einsum('ikl,jkl->ijkl',test,test).transpose(2,3,0,1)
    #
    # S = I_x@I_x.T
    #
    # something = test@I_N
    # something = np.einsum('ijkl,ijkl->ijk',something,something)
    # something = np.sum(something,axis=2).real
    #
    # plt.imshow(something.real)
    # plt.show()
    #
    # exit()
    #
    # b = np.zeros((3,3,len(test)))
    # for i in range(len(test)):
    #     tmp = test[i].reshape(3,1)
    #     tmp = (tmp@tmp.T.conj()).real
    #     b[:,:,i] = tmp
    #
    # test = np.matrix.trace(b)
    #
    # print(b.shape)
    #
    # exit()
    #
    #
    # G_x = G[:,0]
    # G_y = G[:,1]
    # G_z = G[:,2]
    #
    # A_x = np.zeros((N_sensors,N_sensors,N_recon,N_recon),dtype=np.complex128)
    # for i in range(N_recon):
    #     for j in range(N_recon):
    #         A_x[:,:,i,j] = G_x[:,i,j].reshape(-1,1)@G_x[:,i,j].reshape(1,-1).conj()
    #
    # A_x = A_x.transpose(2,3,0,1)
    #
    # P_fov_1 = np.matmul(np.conjugate(A_x),I_N)
    # P_fov_2 = np.matmul(A_x,np.conjugate(I_N))
    #
    # P_fov = np.einsum('ijkl,ijkl->ijk',P_fov_1,P_fov_2)
    # P_fov = np.sum(P_fov.real,axis=2)
    #
    # plt.imshow(1/P_fov)
    # plt.show()
    # exit()
