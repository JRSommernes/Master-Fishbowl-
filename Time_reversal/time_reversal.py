import numpy as np
from constants import *
import matplotlib.pyplot as plt
from numba import jit, void, cuda, vectorize, guvectorize
import sys

# TPB = 3
#
# @cuda.jit
# def fast_conj(E,k_xx,k_yy,k_zz,E_tot):
#     sE = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
#     sxx = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

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
        self.E_tot = np.zeros((len(self.dipoles),3,size,size,size),dtype=np.complex128)

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
            k_x = k_0*sensor.x/sensor.radius
            k_y = k_0*sensor.y/sensor.radius
            k_z = k_0*sensor.z/sensor.radius
            E = np.array(sensor.E)[:,:,np.newaxis,np.newaxis,np.newaxis]
            self.E_tot += sensor.reconstruction(xx*k_x,yy*k_y,zz*k_z,E)
        self.E_tot = np.sum(self.E_tot,axis=0)



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

        self.E.append(G@pol)

    @staticmethod
    # @jit(nopython=True)
    # @cuda.jit('void(float64[:], float64[:], float64[:], complex128[:])')
    # @jit(target ="cuda")
    @vectorize(['complex128(float64, float64, float64, complex128)'],target='parallel')
    # @guvectorize(["complex128(float64[:,:,:],float64[:,:,:],float64[:,:,:],complex128[:,:,:,:,:])"], "(n,n,n),(n,n,n),(n,n,n),(n,n,n,n,n) -> (n,n,n,n)",target="parallel",nopython=True)
    def reconstruction(k_xx,k_yy,k_zz,E):
        E_tot = np.conj(E*np.exp(1j*(k_xx+k_yy+k_zz)))
        # E_tot = np.sum(E_tot,axis=0)
        return E_tot

    # @staticmethod
    # def reconstruction(k_xx,k_yy,k_zz,E):
    #     tmp = np.conj(E*np.exp(1j*(k_xx*k_yy+k_zz)))
    #     tmp = np.sum(tmp,axis=0)
    #
    #     E_global_mem = cuda.to_device(E)
    #     k_xx_global_mem = cuda.to_device(k_xx)
    #     k_yy_global_mem = cuda.to_device(k_yy)
    #     k_zz_global_mem = cuda.to_device(k_zz)
    #     E_tot_global_mem = cuda.device_array((3,100,100,100))
    #
    #     threadsperblock = (TPB, 2*TPB, 2*TPB, 2*TPB)
    #     blockspergrid_x = int(math.ceil(E.shape[0] / threadsperblock[0]))
    #     blockspergrid_y = int(math.ceil(E.shape[1] / threadsperblock[1]))
    #     blockspergrid_z = int(math.ceil(E.shape[2] / threadsperblock[2]))
    #     blockspergrid_k = int(math.ceil(E.shape[3] / threadsperblock[3]))
    #
    #     fast_conj[blockspergrid, threadsperblock](E_global_mem, k_xx_global_mem, k_yy_global_mem, k_zz_global_mem, E_tot_global_mem)
    #     res = E_tot_global_mem.copy_to_host()
    #
    #
    #     exit()
    #     return E_tot
