import numpy as np
from misc_functions import dyadic_green, make_sensors

#Slower if njit because of dyadic green dependecy
def sensor_field(sensors,dipoles,polarizations,N_sensors,k_0):
    E_tot = np.zeros((3*N_sensors,polarizations.shape[2]),dtype=np.complex128)

    for i in range(len(dipoles)):
        G = dyadic_green(sensors,dipoles[i],N_sensors,k_0).reshape(3*N_sensors,3)
        E_tot += G@polarizations[i]

    return E_tot

#Slower if njit because dependencies
def data_acquisition(dipoles,wl,M_inputs,sensor_radius,N_sensors,k_0):
    N_dipoles = len(dipoles)

    phi = np.random.uniform(0,2*np.pi,(N_dipoles,M_inputs))
    theta = np.random.uniform(-np.pi/2,np.pi/2,(N_dipoles,M_inputs))

    polarizations = np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]).swapaxes(0,1)

    sensors = make_sensors(N_sensors,sensor_radius)

    return sensor_field(sensors,dipoles,polarizations,N_sensors,k_0)
