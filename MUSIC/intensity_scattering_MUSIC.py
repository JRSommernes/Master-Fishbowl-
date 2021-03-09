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

def scatter_MUSIC(I,sensors,emitters,N_recon,FoV,k_0,E_sensors):
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

    b = ((I[:,M:2*M-1]-I[:,0].reshape(N_sensors,1)-I[:,1:M])/2).astype(np.complex128)
    b += 1j*(I[:,2*M-1:3*M-2]-I[:,0].reshape(N_sensors,1)-I[:,1:M])/2
    b_0 = I[:,0].reshape(N_sensors,1)
    B = np.append(b_0,b,axis=1).T

    # A = np.zeros((M,K),dtype=np.complex128)
    # for m in range(M):
    #     emitter_pos = emitters[m]
    #     tmp = free_space_green(xx,yy,z,emitter_pos,k_0)[:,0]
    #     A[m] = tmp
    A = np.zeros((M,K),dtype=np.complex128)
    for i,emitter in enumerate(emitters):
        A[i] = incident_wave(xx,yy,z,emitter,k_0).flatten()



    LAM = np.zeros((N_sensors,K),dtype=np.complex128)
    u_1_s = E_sensors[0,0].conj()
    for k in range(K):
        LAM[:,k] = scattered_field_calc(sensors,plane[k],k_0)*u_1_s

    wl = 690e-9
    # pos = np.array([[0*wl, 1*wl, 0*wl]])
    pos = np.array([[0*wl,-1*wl,0*wl],[0*wl,1*wl,0*wl]])
    # pos = np.array([[0*wl, 0*wl, 0*wl]])
    X = np.zeros(len(plane))
    for i in range(len(pos)):
        x_pos = np.argmin(np.abs(x-pos[i,0]))
        y_pos = np.argmin(np.abs(y-pos[i,1]))
        z_pos = np.argmin(np.abs(z-pos[i,2]))
        support = N_recon*x_pos+y_pos
        X[support] = 1

    test_out = np.zeros_like(B)
    for n in range(N_sensors):
        test_out[:,n] = A@np.diag(LAM[n])@X

    # print(E_sensors[0,0].conj()*E_sensors[0,:])
    # print(B[:,0])
    print(((E_sensors[0,0].conj()*E_sensors[0,:M]).imag-B[:,0].imag)/B[:,0].imag)
    exit()

    # print(A.reshape(-1, A.shape[-1]).T.shape,(np.diag(LAM[1])@X).shape)
    # print(A.reshape(-1, A.shape[-1]).T[:,np.where(np.diag(LAM[1])@X!=0)[0]],np.diag(LAM[1])@X)
    # print(x_pos,y_pos)
    # print(A[x_pos,y_pos])

    # print(test_out[0],B[0])
    # test_out = np.zeros((N_sensors,K,3),dtype = np.complex128)
    # for n in range(N_sensors):
    #     for i in range(3):
    #         test_out[n,:,i] = np.diag(LAM[n,:,i])@X


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
