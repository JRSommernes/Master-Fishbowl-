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

def scatter_MUSIC(I,sensors,emitters,N_recon,FoV,k_0,E_sensors,dipoles):
    N_sensors = sensors.shape[1]
    I = I[:N_sensors]

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


    #Construct LAM_n matrix for all n
    LAM = np.zeros((N_sensors,K),dtype=np.complex128)
    for k in range(K):
        LAM[:,k] = scattered_field_calc(sensors,plane[k],k_0)*E_sensors[:,0].conj()

    #Construct the support of X
    X = np.zeros(len(plane))
    for i in range(len(dipoles)):
        x_pos = np.argmin(np.abs(x-dipoles[i,0]))
        y_pos = np.argmin(np.abs(y-dipoles[i,1]))
        z_pos = np.argmin(np.abs(z-dipoles[i,2]))
        support = x_pos+N_recon*y_pos
        X[support] = 0.33

    test_out = np.zeros_like(B)
    # test_out = np.zeros((K,M),dtype=np.complex128)
    for n in range(N_sensors):
        test_out[:,n] = A@np.diag(LAM[n])@X


    # print(np.allclose(test_out[:,0],B[:,0]))
    # print(np.allclose(((E_sensors[0,0].conj()*E_sensors[0,:M])),B[:,0]))
    # exit()


    U,Sigma,V = np.linalg.svd(B)

    NN = np.where(Sigma>0.1*np.max(Sigma))[0]
    U_tilde = U[:,NN]

    P = np.identity(len(U_tilde)) - U_tilde@U_tilde.T.conj()


    # NN = np.where(Sigma>0.1*np.max(Sigma))[0]
    # V_tilde = V[:,NN]
    #
    # P = np.identity(len(V_tilde)) - V_tilde@V_tilde.T.conj()
    #
    # plt.imshow(np.abs(np.sum((P@test_out),axis=0).reshape((N_recon,N_recon))))
    # plt.show()


    njy = np.sum(P@A,axis=0)
    # njy = np.sum(np.einsum('ij,kli->klj',P,A),axis=2)
    njy_min = np.min(njy)

    Im = (njy_min/njy).reshape((N_recon,N_recon))
    plt.imshow(np.abs(Im))
    plt.show()

################################################################################

    # wrong = np.where(Im==np.max(Im))
    # wrong = np.array((wrong[0][0]))
    #
    # wl = 690e-9
    # err = np.sum((plane-np.array([[1*wl, 0*wl,0*wl]]).T)**2,axis=0)
    # col = int(np.argmin(err)%N_recon)
    # row = int((np.argmin(err)-np.argmin(err)%N_recon)/N_recon)
    # pos = np.array((x[col],y[row],0))
    #
    # print(A[:,col//1*N_recon+row])
    #
    # test = np.zeros(len(A),dtype=np.complex128)
    # for i in range(len(emitters)):
    #     test[i] = free_space_green(pos.reshape(3,1),emitters[i],k_0)[:,0]




    # print(np.argmin(err))
    # print(wrong)