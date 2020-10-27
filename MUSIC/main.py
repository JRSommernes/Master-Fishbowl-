import numpy as np
import matplotlib.pyplot as plt
from time import time
from MUSIC import *
from misc_functions import plot_sensors


eps_0 = 8.8541878128e-12
mu_0 = 4*np.pi*10**-7
c_0 = 1/np.sqrt(eps_0*mu_0)

wl = 690e-9
freq = c_0/wl
k_0 = 2*np.pi*freq*np.sqrt(eps_0*mu_0)
omega = 2*np.pi*freq

sensor_radius = 10*wl
N_sensors = 300
M_inputs = 100

dipoles = np.array([[0.7*wl,0,0]])

phi = np.linspace(0,2*np.pi,M_inputs)
theta = np.linspace(-np.pi/2,np.pi/2,M_inputs)

polarizations = []
for i in range(M_inputs):
    polarizations.append([np.cos(phi[i])*np.sin(theta[i]),np.sin(phi[i])*np.sin(theta[i]),np.cos(theta[i])])
polarizations = np.array(polarizations)

# dipoles = np.array([[0.7*wl,0,0]])
#
# phi_1, theta_1 = np.pi, np.pi/4
#
# polarizations = np.array([[np.cos(phi_1)*np.sin(theta_1),np.sin(phi_1)*np.sin(theta_1),np.cos(theta_1)]])

sensors = make_sensors(N_sensors,sensor_radius)
plot_sensors(sensors,wl)

E_sensors = sensor_field(sensors,dipoles,polarizations,N_sensors,k_0)

# reconstruct(E_sensors)
for i in range(4):
    t0 = time()
    something(E_sensors,N_sensors,sensor_radius,k_0,wl)
    print(time()-t0)





I = np.sqrt(np.sum(E_sensors**2,axis=1))
