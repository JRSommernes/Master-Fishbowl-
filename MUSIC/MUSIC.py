import numpy as np
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython=True, parallel=True)
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

@jit(nopython=True, parallel=True)
def dyadic_green(sensors,dipole_pos,N_sensors,k_0):
    # x,y,z = dipole_pos
    # r_p = np.array([sensors[0]-x, sensors[1]-y, sensors[2]-z])
    r_p = sensors-dipole_pos.reshape(len(dipole_pos),1)

    R = np.sqrt(np.sum((r_p)**2,axis=0))
    R_hat = ((r_p)/R)

    RR_hat = np.zeros((3,3,N_sensors))

    for i in range(N_sensors):
        RR_hat[:,:,i] = R_hat[:,i].copy().reshape(3,1)@R_hat[:,i].copy().reshape(1,3)

    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
    expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
    expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R

    I = np.identity(3)
    # G = (expr_1*RR_hat + expr_2*I[:,:,np.newaxis]).T
    G = (expr_1*RR_hat + expr_2*I.reshape(3,3,1)).T

    return G

# @jit(nopython=True, parallel=True)
def sensor_field(sensors,dipoles,polarizations,N_sensors,k_0):
    E_tot = np.zeros((N_sensors,3),dtype=np.complex128)

    for i in range(len(dipoles)):
        G = dyadic_green(sensors,dipoles[i],N_sensors,k_0)
        E_tot += G.dot(polarizations[i])

    return E_tot

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
