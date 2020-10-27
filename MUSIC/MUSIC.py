import numpy as np
from numba import njit, guvectorize, float64, complex128, int32
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from time import time

@njit
def make_sensors(N_sensors,sensor_radius):
    sensors = np.zeros((N_sensors,3))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(N_sensors):
        y = (1 - (i / float(N_sensors - 1)) * 2)  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        x,y,z = x*sensor_radius, y*sensor_radius, z*sensor_radius

        sensors[i] = [x,y,z]
    return sensors.T

@njit
def dyadic_green(sensors,dipole_pos,N_sensors,k_0):
    r_p = sensors-dipole_pos.reshape(3,1)

    R = np.sqrt(np.sum((r_p)**2,axis=0))
    R_hat = ((r_p)/R)

    RR_hat = np.zeros((3,3,N_sensors))

    for i in range(N_sensors):
        RR_hat[:,:,i] = R_hat[:,i].copy().reshape(3,1)@R_hat[:,i].copy().reshape(1,3)

    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
    expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
    expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R

    I = np.identity(3)
    G = (expr_1*RR_hat + expr_2*I.reshape(3,3,1)).T

    return G

# @jit(nopython=True, parallel=True)
def sensor_field(sensors,dipoles,polarizations,N_sensors,k_0):
    E_tot = np.zeros((3*N_sensors,len(polarizations)),dtype=np.complex128)

    for i in range(len(dipoles)):
        G = dyadic_green(sensors,dipoles[i],N_sensors,k_0).reshape(3*N_sensors,3)
        E_tot += G@polarizations.T

    return E_tot

def reconstruct(E_field):
    S = np.conjugate(E_field@np.conjugate(E_field).T)
    M = S.shape[0]

    eigvals,eigvecs = np.linalg.eig(S)
    mat = np.concatenate((eigvals.real.reshape(M,1),eigvals.imag.reshape(M,1)),axis=1)

    min = np.min(eigvals)
    min_idx = np.where(eigvals==min)[0][0]
    dist = cdist(mat,mat)[min_idx]

    noice_idx = np.where(dist<3e-4)[0]
    N = len(noice_idx)
    D = len(E_field)-N

    E_N = eigvecs[noice_idx]

    return E_N.T@np.conjugate(E_N)

# @njit(parallel=True)
def something(E_field,N_sensors,sensor_radius,k_0,wl):

    sensors = make_sensors(N_sensors,sensor_radius)
    # E_N = reconstruct(E_field)

    a = 10
    x = np.linspace(-2*wl,2*wl,a)
    y = np.linspace(-2*wl,2*wl,a)
    z = np.linspace(-2*wl,2*wl,a)

    # P_theta = np.zeros((a,b,c,3,3),dtype=np.complex128)
    for i in range(a):
        for j in range(a):
            for k in range(a):
                x_i,y_i,z_i = x[i],y[i],z[i]
                pos = np.array((x_i,y_i,z_i))
                A = dyadic_green(sensors,pos,N_sensors,k_0)

                # P_theta[i,j,k] = 1/(np.conjugate(A).T@E_N@A)

    # P_ish = np.sum(np.sum(P_theta,axis=4),axis=3)
    #
    # print(np.where(P_ish == np.amax(P_ish)))
    #
    # plt.imshow(np.abs(P_ish[:,:,8]))
    # plt.show()


    # G = np.zeros((N_sensors,3,3),dtype=np.complex128)
    # for dipole_pos in dipoles:
    #     G += dyadic_green(sensors,dipole_pos,N_sensors,k_0)
    # G = G.reshape((3*N_sensors,3))

    # tmp = np.array([[np.cos( np.pi)*np.sin(np.pi/4),np.sin( np.pi)*np.sin(np.pi/4),np.cos(np.pi/4)]]).T
    # print(X-G@tmp)


    #NOPE
    # p = np.linspace(0,2*np.pi,20)
    # t = np.linspace(-np.pi/2,np.pi/2,15)
    #
    # phi,theta = np.meshgrid(p,t)
    #
    # a_x = (np.cos(phi)*np.sin(theta)).reshape(len(E_field),1)
    # a_y = (np.sin(phi)*np.sin(theta)).reshape(len(E_field),1)
    # a_z = (np.cos(theta)).reshape(len(E_field),1)
    #
    # a = np.append(np.append(a_x,a_y,axis=1),a_x,axis=1)
    #
    # test = 1/(np.conjugate(a).T@E_N@np.conjugate(E_N).T@a)
    #
    # print(test)





# def sensor_field_time(sensors,dipoles,polarizations,N_sensors,k_0,t,omega):
#     E_tot = np.zeros((N_sensors,3,len(t)),dtype=np.complex128)
#     for i in range(len(dipoles)):
#         G = dyadic_green(sensors,dipoles[i],N_sensors,k_0)
#         E_sensors = G.dot(polarizations[i])
#         dists = np.sqrt(np.sum((sensors-dipoles[i].reshape(3,1))**2,axis=0))
#         E = E_sensors*((np.exp(-1j*omega*(dists/299792458)).reshape(N_sensors,1)).reshape(N_sensors,1))
#         E_time = (E.reshape(N_sensors,3,1)*np.exp(-1j*omega*t))*np.exp(-1j*k_0*dists)[:,np.newaxis,np.newaxis]
#         #THIS IS FUCK, DIST=300, t=1000
#         E_tot += E_time
#
#     return E_tot
