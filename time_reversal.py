import numpy as np
from constants import *
import matplotlib.pyplot as plt
from numba import jit, void
import sys

class Microscope:
    def __init__(self,N_sensors,N_reconstruction,FoV):
        self.sensor_ammount = N_sensors
        self.reconstruction_size = N_reconstruction
        self.FoV = FoV

    def make_dipoles(self,dipole_pos,pol):
        dipoles = []
        for i,dipole in enumerate(dipole_pos):
            x,y,z = dipole
            x_pol, y_pol, z_pol = pol[i]
            dipoles.append(Dipole(x, y, z, x_pol, y_pol, z_pol))
        self.dipoles = dipoles

    def make_sensors(self,sensor_radius):
        sensors = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        N = self.sensor_ammount

        for i in range(N):
            y = (1 - (i / float(N - 1)) * 2)  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            x,y,z = x*sensor_radius, y*sensor_radius, z*sensor_radius

            sensors.append(Sensor(x,y,z,sensor_radius))
        self.sensors = sensors

    def record_sensors(self):
        for sensor in self.sensors:
            for dipole in self.dipoles:
                pol = np.array([dipole.x_pol,dipole.y_pol,dipole.z_pol])
                sensor.dipole_field(dipole.x,dipole.y,dipole.z,pol)

    def reconstruct_image(self,size):
        self.record_sensors()
        self.E_tot = np.zeros((3,size,size,size),dtype=np.complex128)

        x = np.linspace(-self.FoV/2,self.FoV/2,self.reconstruction_size)
        y = np.linspace(-self.FoV/2,self.FoV/2,self.reconstruction_size)
        z = np.linspace(-self.FoV/2,self.FoV/2,self.reconstruction_size)
        # x = np.linspace(0,FoV/2,N_reconstruction)
        # y = np.linspace(0,FoV/2,N_reconstruction)
        # z = np.linspace(0,FoV/2,N_reconstruction)
        xx,yy,zz = np.meshgrid(x,y,z)

        counter=0
        for sensor in self.sensors:
            counter+=1
            if counter%(self.sensor_ammount//100)==0:
                done = (counter*100)//self.sensor_ammount
                # print('{} %   '.format(done), end="\r")
                sys.stdout.write('\r')
                sys.stdout.write("[%-100s] %d%%" % ('='*done, done))
                sys.stdout.flush()
            self.E_tot += sensor.reconstruction(self.reconstruction_size,xx,yy,zz)
        print("\n")

        self.I = np.sqrt((np.abs(self.E_tot[0])**2)+(np.abs(self.E_tot[1])**2)+(np.abs(self.E_tot[2])**2))



class Dipole:
    def __init__(self, x, y, z, x_pol, y_pol, z_pol):
        self.x = x
        self.y = y
        self.z = z
        self.x_pol = x_pol
        self.y_pol = y_pol
        self.z_pol = z_pol

class Sensor:
    def __init__(self, x, y, z, sensor_radius):
        self.x = x
        self.y = y
        self.z = z
        self.radius = sensor_radius

        self.E = []
        self.time_lag = []

    def find_time_lag(self,r):
        dist = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
        self.time_lag.append(dist/c_0)

    def dipole_field(self,x,y,z,pol):
        r_x, r_y, r_z = self.x-x, self.y-y, self.z-z

        r_p = np.array((r_x,r_y,r_z))

        self.find_time_lag(r_p)

        R = np.sqrt(np.sum((r_p)**2))
        R_hat = ((r_p)/R)

        RR_hat = R_hat.reshape(3,1).dot(R_hat.reshape(1,3))

        g_R = np.exp(1j*k_0*R)/(4*np.pi*R)
        expr_1 = (3/(k_0**2*R**2)-3j/(k_0*R)-1)*g_R
        expr_2 = (1+1j/(k_0*R)-1/(k_0**2*R**2))*g_R

        I = np.identity(3)
        G = (expr_1*RR_hat + expr_2*I).T

        self.E.append(np.dot(G,pol))

    def reconstruction(self,N,xx,yy,zz):
        k_x = k_0*self.x/self.radius
        k_y = k_0*self.y/self.radius
        k_z = k_0*self.z/self.radius
        E_tot = np.zeros(([3,N,N,N]),dtype=np.complex128)
        for i,E in enumerate(self.E):
            E_tot[0] += np.conj(E[0]*np.exp(1j*(k_x*xx+k_y*yy+k_z*zz)))
            E_tot[1] += np.conj(E[1]*np.exp(1j*(k_x*xx+k_y*yy+k_z*zz)))
            E_tot[2] += np.conj(E[2]*np.exp(1j*(k_x*xx+k_y*yy+k_z*zz)))
            #*np.exp(-1j*omega*self.time_lag[i])

        return E_tot
