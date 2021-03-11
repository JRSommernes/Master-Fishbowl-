import numpy as np
try:
    import cupy as cp
except:
    pass
from time import time
from misc_functions import dyadic_green, high_inner, loadbar
import matplotlib.pyplot as plt

def incident_wave(xx,yy,z,emitter_pos,k_0):
    r_x = xx.reshape(*xx.shape)-emitter_pos[0]
    r_y = yy.reshape(*yy.shape)-emitter_pos[1]
    r_z = z-emitter_pos[2]

    R = np.sqrt(r_x**2+r_y**2+r_z**2)
    g_R = np.exp(1j*k_0*R)

    return g_R

def incident_wave_sensors(scatterers,emitter_pos,k_0):
    r_p = scatterers-emitter_pos.reshape(3,1)
    R = np.sqrt(np.sum((r_p)**2,axis=0))

    g = np.exp(1j*k_0*R)
    return g

def scattered_field_calc(sensors,scatterer,k_0):
    r_p = sensors-scatterer.reshape(3,1)
    R = np.sqrt(np.sum((r_p)**2,axis=0))
    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)

    return g_R

def free_space_green_sensors(sensors,emitter_pos,k_0):
    r_p = sensors-emitter_pos.reshape(3,1)
    R = np.sqrt(np.sum((r_p)**2,axis=0))
    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)

    return g_R

def scatter_MUSIC(I,sensors,emitters,N_recon,FoV,k_0,E_scatter,E_incident,dipoles):
    N_sensors = sensors.shape[1]
    I_x = I[:N_sensors]
    I_y = I[N_sensors:2*N_sensors]
    I_z = I[2*N_sensors:]
    for I in (I_x,I_y,I_z):
        x = np.linspace(FoV[0,0],FoV[0,1],N_recon)
        y = np.linspace(FoV[1,0],FoV[1,1],N_recon)
        # z = np.linspace(FoV[2,0],FoV[2,1],N_recon)
        z = 0

        xx,yy = np.meshgrid(x,y)

        plane = np.array((xx.flatten(),yy.flatten(),np.zeros_like(xx.flatten()))).T

        K = xx.size
        M = np.shape(emitters)[0]

        b = (I[:,M:2*M-1]-I[:,0].reshape(N_sensors,1)-I[:,1:M])/2 \
            - 1j*(I[:,2*M-1:]-I[:,0].reshape(N_sensors,1)-I[:,1:M])/2
        b_0 = I[:,0].reshape(N_sensors,1)
        B = np.append(b_0,b,axis=1).T

        #Construct the A matrix
        A = np.zeros((M,K),dtype=np.complex128)
        for i,emitter in enumerate(emitters):
            A[i] = incident_wave(xx,yy,z,emitter,k_0).flatten()

    E_i_m = np.zeros_like(B)
    for m in range(M):
        E_i_m[m] = incident_wave_sensors(sensors,emitters[m],k_0)

    E_i_1 = E_i_m[0]
    E_i = E_i_m@np.diag(E_i_1.conj())
    B -= E_i

    E_i_E_s = E_scatter[:,:M].T@np.diag(E_i_1.conj())
    B -= E_i_E_s

    E_s_E_i = E_i_m@np.diag(E_scatter[:,0].conj())
    B -= E_s_E_i

    E_s = E_scatter[:,:M].T@np.diag(E_scatter[:,0].conj())
    B -= E_s

    print(B[1,1])

    #
    # test_1 = np.array(((1,1,1),(1,1,1),(1,1,1)))
    # test_2 = np.diag(np.array((1,2,3)))
    # print(test_1@test_2)
    exit()

    #Construct the A matrix
    A = np.zeros((M,K),dtype=np.complex128)
    for i,emitter in enumerate(emitters):
        A[i] = incident_wave(xx,yy,z,emitter,k_0).flatten()

        # #Construct LAM_n matrix for all n
        # LAM = np.zeros((N_sensors,K),dtype=np.complex128)
        # for k in range(K):
        #     LAM[:,k] = scattered_field_calc(sensors,plane[k],k_0)*E_sensors[:N_sensors,0].conj()
        #
        # #Construct the support of X
        # X = np.zeros(len(plane))
        # for i in range(len(dipoles)):
        #     x_pos = np.argmin(np.abs(x-dipoles[i,0]))
        #     y_pos = np.argmin(np.abs(y-dipoles[i,1]))
        #     z_pos = np.argmin(np.abs(z-dipoles[i,2]))
        #     support = x_pos+N_recon*y_pos
        #     X[support] = 0.33
        #
        # test_out = np.zeros_like(B)
        # for n in range(N_sensors):
        #     test_out[:,n] = A@np.diag(LAM[n])@X

    #Construct LAM_n matrix for all n
    LAM = np.zeros((N_sensors,K),dtype=np.complex128)
    for k in range(K):
        LAM[:,k] = scattered_field_calc(sensors,plane[k],k_0)*E_scatter[:,0].conj()

        # print(np.allclose(test_out[:,0],B[:,0]))
        # print(np.allclose(((E_sensors[0,0].conj()*E_sensors[0,:M])),B[:,0]))
        # exit()


        print(B.shape)
        exit()

    # print(np.allclose(test_out[:,0],B[:,0]))
    # print(np.allclose(((E_scatter[0,0].conj()*E_scatter[0,:M])),B[:,0]))
    # exit()

        NN = np.where(Sigma>0.1*np.max(Sigma))[0]
        U_tilde = U[:,NN]

        P = np.identity(len(U_tilde)) - U_tilde@U_tilde.T.conj()


        njy = np.sum(P@A,axis=0)
        njy_min = np.min(njy)

        Im = (njy_min/njy).reshape((N_recon,N_recon))
        plt.imshow(np.abs(Im))
        plt.show()
