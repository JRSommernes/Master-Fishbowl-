import numpy as np
from misc_functions import dyadic_green, scalar_green, plot_sensors

def incident_wave(scatterers,emitter_pos,k_0):
    r_p = scatterers-emitter_pos.reshape(3,1)
    R = np.sqrt(np.sum((r_p)**2,axis=0))

    g = np.exp(1j*k_0*R)
    return g

def scattered_field_calc(sensors,scatterer,k_0):
    r_p = sensors-scatterer.reshape(3,1)
    R = np.sqrt(np.sum((r_p)**2,axis=0))
    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)

    return g_R

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

def scattered_field(dipoles,emitter_loc,sensors,k_0,n_0,n_1):
    E_i = incident_wave(dipoles.T,emitter_loc,k_0)

    E_s = np.zeros((sensors.shape[1]),dtype=np.complex128)
    for i,dipole in enumerate(dipoles):
        G = scattered_field_calc(sensors,dipole,k_0)
        E_s += G*E_i[i]*(n_1-n_0)

    return E_s

def sensor_field(sensors,emitters,dipoles,N_sensors,N_emitters,k_0,n0,n1):
    M_inputs = 3*N_emitters-2
    E_scatter = np.zeros((N_sensors,M_inputs),dtype=np.complex128)
    E_incident = np.zeros((N_sensors,M_inputs),dtype=np.complex128)

    for i,emitter_loc in enumerate(emitters):
        E_s = scattered_field(dipoles,emitter_loc,sensors,k_0,n0,n1).T.flatten()
        E_i = scattered_field_calc(sensors,emitter_loc,k_0).T.flatten()
        E_scatter[:,i] = E_s
        E_incident[:,i] = E_i

    for i,emitter_loc in enumerate(emitters[1:]):
        E_s_1 = scattered_field(dipoles,emitters[0],sensors,k_0,n0,n1).T.flatten()
        E_s_n = scattered_field(dipoles,emitter_loc,sensors,k_0,n0,n1).T.flatten()
        E_i_1 = scattered_field_calc(sensors,emitters[0],k_0).T.flatten()
        E_i_n = scattered_field_calc(sensors,emitter_loc,k_0).T.flatten()
        E_scatter[:,N_emitters+i] = E_s_1+E_s_n
        E_incident[:,N_emitters+i] = E_i_1+E_i_n


    for i,emitter_loc in enumerate(emitters[1:]):
        E_s_1 = scattered_field(dipoles,emitters[0],sensors,k_0,n0,n1).T.flatten()
        E_s_n = scattered_field(dipoles,emitter_loc,sensors,k_0,n0,n1).T.flatten()
        E_i_1 = scattered_field_calc(sensors,emitters[0],k_0).T.flatten()
        E_i_n = scattered_field_calc(sensors,emitter_loc,k_0).T.flatten()
        E_scatter[:,2*N_emitters-1+i] = E_s_1+1j*E_s_n
        E_incident[:,2*N_emitters-1+i] = E_i_1+1j*E_i_n


    return E_scatter,E_incident

def scattering_data(dipoles,sensor_radius,N_sensors,N_emitters,k_0,n1,n0):
    N_dipoles = len(dipoles)
    M_inputs = 3*N_emitters-2

    sensors = make_sensors(N_sensors,sensor_radius)
    emitters = make_emitters(N_emitters,sensor_radius)

    E_scatter, E_incident = sensor_field(sensors,emitters,dipoles,N_sensors,N_emitters,k_0,n0,n1)

    return E_scatter, E_incident, sensors, emitters
