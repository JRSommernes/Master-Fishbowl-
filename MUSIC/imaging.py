import numpy as np
import csv

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

def dyadic_green(sensors,dipole_pos,N_sensors,k_0):
    r_p = sensors-dipole_pos.reshape(3,1)

    R = np.sqrt(np.sum((r_p)**2,axis=0))
    R_hat = ((r_p)/R)

    RR_hat = np.zeros((3,3,N_sensors))

    for i in range(N_sensors):
        RR_hat[:,:,i] = R_hat[:,i].copy().reshape(3,1)@R_hat[:,i].copy().reshape(1,3)

    print(RR_hat[:,:,0])

    g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
    expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
    expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R

    I = np.identity(3)
    G = (expr_1*RR_hat + expr_2*I.reshape(3,3,1)).T

    return G

def sensor_field(sensors,dipoles,polarizations,N_sensors,k_0):
    E_tot = np.zeros((3*N_sensors,len(polarizations)),dtype=np.complex128)

    for i in range(len(dipoles)):
        G = dyadic_green(sensors,dipoles[i],N_sensors,k_0).reshape(3*N_sensors,3)
        E_tot += G@polarizations.T

    return E_tot

def data_acquisition(dipoles,wl,M_inputs,sensor_radius,N_sensors,k_0,input_file=None):
    if input_file != None:
        phi = []
        theta = []
        f = open('angles.csv', 'r')
        with f:
            reader = csv.DictReader(f)
            for row in reader:
                phi.append(float(row['phi']))
                theta.append(float(row['theta']))
        phi = np.array(phi)
        theta = np.array(theta)

    else:
        phi = np.random.uniform(0,2*np.pi,M_inputs)
        theta = np.random.uniform(-np.pi/2,np.pi/2,M_inputs)
        with open('angles.csv', mode='w') as csv_file:
            fnames = ['phi', 'theta']
            angle_writer = csv.DictWriter(csv_file, fieldnames=fnames)

            angle_writer.writeheader()
            for i in range(len(phi)):
                angle_writer.writerow({'phi' : phi[i], 'theta' : theta[i]})

    polarizations = []
    for i in range(M_inputs):
        polarizations.append([np.cos(phi[i])*np.sin(theta[i]),np.sin(phi[i])*np.sin(theta[i]),np.cos(theta[i])])
    polarizations = np.array(polarizations)

    sensors = make_sensors(N_sensors,sensor_radius)

    return sensor_field(sensors,dipoles,polarizations,N_sensors,k_0)
