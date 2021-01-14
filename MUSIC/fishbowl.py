import numpy as np
from misc_functions import *

def correlation_space(E_field):
    S = E_field@np.conjugate(E_field).T

    eigvals,eigvecs = np.linalg.eig(S)

    dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)

    noice_idx = np.where(dist<1)[0]
    N = len(noice_idx)
    D = len(E_field)-N
    print(D//3)
    #
    # E_N = eigvecs[:,noice_idx]

    return np.ascontiguousarray(eigvecs)

def fishbowl(E_sensors,sensors,wl,k_0,pos):
    wl = 690e-9

    E_x = E_sensors[0::3]
    E_y = E_sensors[1::3]
    E_z = E_sensors[2::3]

    E_sensors = np.append(E_x,np.append(E_y,E_z,axis=0),axis=0)

    theta = np.arctan2(np.sqrt(sensors[0]**2+sensors[1]**2),sensors[2])
    phi = np.arctan2(sensors[1],sensors[0])

    I_x = np.abs(E_x)**2
    I_y = np.abs(E_y)**2
    I_z = np.abs(E_z)**2

    # E_theta = np.arctan2(np.sqrt(I_x**2+I_y**2),I_z)
    # E_phi = np.arctan2(I_y,I_z)

    theta = np.pi/2-theta

    z_cont = np.sin(theta)
    #x and y bit trickyer, theta and phi dependency

    # print(sensors[:,2]/(10*wl),theta[2],phi[2])

    test = np.array([theta,z_cont])
    test = test[:, np.argsort( test[0] ) ]

    plt.plot(test[0],c='r')
    plt.plot(test[1],c='b')
    plt.show()

    # print(sensors[:,0])
    # plt.plot(E_x[0],c='b')
    # plt.plot(E_y[0],c='g')
    # plt.plot(E_z[0],c='y')
    # plt.plot(E_theta[20],c='r')
    # plt.show()
    exit()

    test = correlation_space(E_x)

    N_theta = correlation_space(E_theta)
    N_phi = correlation_space(E_phi)
    exit()




    G = dyadic_green(sensors,pos,k_0)

    A = G#.reshape(-1, G.shape[-1])[0::3]
    B =  N_theta.reshape(-1, N_theta.shape[-1])
    C =  N_phi.reshape(-1, N_phi.shape[-1])

    T_1 = np.conjugate(A.T)@B
    T_2 = np.conjugate(A.T)@C

    T = np.append(T_1,T_2,axis=0)

    # print(A.shape,B.shape)
    # print(T_1.shape)
    # exit()

    return T_1,T_2

    # return T


    # Simple P calulation
    # pos = np.array([[0,0,0]])
    # G = dyadic_green(sensors,pos,k_0)
    #
    # A = G.reshape(-1, G.shape[-1])[0::3]
    # B =  N_theta.reshape(-1, N_theta.shape[-1])
    #
    # P_1 = np.conjugate(A.T)@B
    # P_2 = A.T@np.conjugate(B)
    #
    # P = (1/np.einsum('ij,ij->i',P_1,P_2)).T
    # P = np.sum(P,axis=0)
    #
    # print(P)
    # exit()
