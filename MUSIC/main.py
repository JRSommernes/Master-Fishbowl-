import numpy as np
import matplotlib.pyplot as plt
from MUSIC import *

eps_0 = 8.8541878128e-12
mu_0 = 4*np.pi*10**-7
c_0 = 1/np.sqrt(eps_0*mu_0)

wl = 690e-9
freq = c_0/wl
k_0 = 2*np.pi*freq*np.sqrt(eps_0*mu_0)
omega = 2*np.pi*freq

sensor_radius = 10*wl
N_sensors = 300

dipoles = np.array([[0.5*wl,0,0],[-0.5*wl,0,0]])

phi_1, theta_1 = np.pi, np.pi/4
phi_2, theta_2 = np.pi, np.pi/4

polarizations = np.array([[np.cos(phi_1)*np.sin(theta_1),np.sin(phi_1)*np.sin(theta_1),np.cos(theta_1)]\
                ,[np.cos(phi_2)*np.sin(theta_2),np.sin(phi_2)*np.sin(theta_2),np.cos(theta_2)]])

t = np.linspace(0,2*np.pi/omega,1000)

sensors = make_sensors(N_sensors,sensor_radius)
E_sensors = sensor_field_time(sensors,dipoles,polarizations,N_sensors,k_0,t,omega)

print(E_sensors[0,:,0])

I = np.sqrt(np.sum(E_sensors**2,axis=1))

# for i in range(N_sensors):
#     plt.plot(E_sensors[i])
#     plt.show(block=False)
#     plt.pause(0.1)
#     plt.clf()
#     plt.cla()
