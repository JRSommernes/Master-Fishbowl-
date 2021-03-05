import numpy as np
from misc_functions import dyadic_green, scalar_green, plot_sensors

def free_space_green(sensors,emitter_pos,k_0):
    r_p = sensors-emitter_pos.reshape(3,1)
    R = np.sqrt(np.sum((r_p)**2,axis=0))
    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)

    E = np.array((g_R,np.zeros_like(g_R),np.zeros_like(g_R))).T
    return E

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

def make_emitters(N_emitters,emitter_radius):
    emitters = make_sensors(N_emitters,emitter_radius)
    x = emitters[0]
    y = emitters[1]
    z = emitters[2]

    r = emitter_radius
    theta = np.arctan2(np.sqrt(x**2+y**2),z)
    phi = np.arctan2(y,x)+0.5

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    emitters = np.array((x,y,z))
    return np.ascontiguousarray(emitters.T)

def scattered_field(dipoles,pol,emitter_loc,sensors,k_0,n_0,n_1):
    E_i = free_space_green(dipoles.T,emitter_loc,k_0)

    E_s = np.zeros((sensors.shape[1],3),dtype=np.complex128)
    for i,dipole in enumerate(dipoles):
        G = dyadic_green(sensors,dipole,k_0)
        E_s += G@E_i[i]*(n_1-n_0)

    return E_s

#Slower if njit because of dyadic green dependecy
def sensor_field(sensors,emitters,dipoles,polarizations,N_sensors,N_emitters,k_0,n1,n0):
    M_inputs = 3*N_emitters-2
    E_tot = np.zeros((3*N_sensors,M_inputs),dtype=np.complex128)

    for emitter_loc in emitters:
        E_s = scattered_field(dipoles,polarizations,emitter_loc,sensors,k_0,n1,n0)
        E_i = free_space_green(sensors,emitter_loc,k_0)

    E_tot = E_s+E_i

    return E_tot

#Slower if njit because dependencies
def scattering_data(dipoles,sensor_radius,N_sensors,N_emitters,k_0,n1,n0):
    N_dipoles = len(dipoles)
    M_inputs = 3*N_emitters-2

    phi = np.random.uniform(0,2*np.pi,(N_dipoles,M_inputs))
    theta = np.random.uniform(-np.pi/2,np.pi/2,(N_dipoles,M_inputs))

    phi = np.ones_like(phi)
    theta = np.ones_like(theta)

    polarizations = np.array([np.cos(phi)*np.sin(theta),
                              np.sin(phi)*np.sin(theta),
                              np.cos(theta)]).swapaxes(0,1)


    sensors = make_sensors(N_sensors,sensor_radius)
    emitters = make_emitters(N_emitters,sensor_radius)

    E_sensors = sensor_field(sensors,emitters,dipoles,polarizations,N_sensors,N_emitters,k_0,n1,n0)

    return E_sensors, sensors, emitters
