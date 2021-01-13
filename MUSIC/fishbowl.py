import numpy as np
from misc_functions import *

def correlation_space(E_field):
    S = E_field@np.conjugate(E_field).T

    eigvals,eigvecs = np.linalg.eig(S)

    # dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)
    #
    # noice_idx = np.where(dist<1)[0]
    # N = len(noice_idx)
    # D = len(E_field)-N
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

    E_theta = np.arctan2(np.sqrt(I_x**2+I_y**2),I_z)
    E_phi = np.arctan2(I_y,I_z)

    N_theta = correlation_space(E_theta)
    N_phi = correlation_space(E_phi)




    G = dyadic_green(sensors,pos,k_0)

    A = G.reshape(-1, G.shape[-1])[0::3]
    B =  N_theta.reshape(-1, N_theta.shape[-1])
    C =  N_phi.reshape(-1, N_phi.shape[-1])

    T_1 = np.conjugate(A.T)@B
    T_2 = np.conjugate(A.T)@C

    T = np.append(T_1,T_2,axis=0)

    return T


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
