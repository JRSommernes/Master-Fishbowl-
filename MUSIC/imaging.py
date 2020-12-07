import numpy as np
from misc_functions import dyadic_green

#Faster if run thousand times with njit
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
    return np.ascontiguousarray(sensors.T)

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

    polarizations = np.array([np.cos(phi)*np.sin(theta),
                              np.sin(phi)*np.sin(theta),
                              np.cos(theta)]).swapaxes(0,1)

    sensors = make_sensors(N_sensors,sensor_radius)

    E_sensors = sensor_field(sensors,dipoles,polarizations,N_sensors,k_0)

    return E_sensors, sensors
