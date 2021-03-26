import numpy as np
from scipy.constants import epsilon_0, mu_0, c
from time import time
from scipy.integrate import quad_vec
from scipy.special import jv
import matplotlib.pyplot as plt
from numba import jit
import sys, os, json
from PIL import Image

def plot_sensors(sensors,wl):
    coordinates = sensors/wl
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = coordinates[0],coordinates[1],coordinates[2]
    scat = ax.scatter(X, Y, Z)

    ax.set_box_aspect([X.max(),Y.max(),Z.max()])
    ax.set_xlabel('x-position [wavelengths]')
    ax.set_ylabel('y-position [wavelengths]')
    ax.set_zlabel('z-position [wavelengths]')
    # plt.savefig('Detector_locations')
    plt.show()

class Fishbowl:
    def __init__(self,N_sensors,radius,wl,n,mur,epsr,k_0,NA,z_sub,dipoles,M_timepoints):
        self.N_sensors = N_sensors
        self.radius = radius
        self.wl = wl
        self.n_obj, self.n_sub = n
        self.mur_obj, self.mur_sub = mur
        self.epsr_obj, self.epsr_sub = epsr
        self.k_0 = k_0
        self.z_sub = z_sub
        self.dipoles = dipoles
        self.M_timepoints = M_timepoints

        self.k_obj = self.k_0*self.n_obj
        self.k_sub = self.k_0*self.n_sub

    def make_bowl_sensors(self):
        sensors = np.zeros((self.N_sensors,3))
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

        for i in range(self.N_sensors):
            y = (1 - (i / float(self.N_sensors - 1)) * 2)  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            x,y,z = x*self.radius, y*self.radius, z*self.radius

            sensors[i] = [x,y,z]
        self.sensors = np.ascontiguousarray(sensors.T)

    def limited_aperture_sensors(self,NA):
        self.NA = NA
        self.theta_max = np.arcsin(self.NA/self.n_obj)

        sphere_area = 4*np.pi*self.radius**2
        cap_area = 2*np.pi*self.radius**2*(1-np.cos(self.theta_max))
        mult = sphere_area/cap_area
        N_sensors = self.N_sensors
        self.N_sensors = int(self.N_sensors*mult)

        while 1:
            self.make_bowl_sensors()
            theta = np.arctan2(np.sqrt(self.sensors[0]**2+self.sensors[1]**2),self.sensors[2])
            if len(np.where(theta<=self.theta_max)[0]) > N_sensors:
                self.N_sensors -= 1
            elif len(np.where(theta<=self.theta_max)[0]) < N_sensors:
                self.N_sensors += 1
            else:
                break

        idx = np.where(theta<=self.theta_max)[0]
        self.sensors = self.sensors[:,idx]
        self.N_sensors = N_sensors

    def dyadic_green(self,dipole_pos):
        r_p = self.sensors-dipole_pos.reshape(3,1)

        R = np.sqrt(np.sum((r_p)**2,axis=0))
        R_hat = ((r_p)/R)

        RR_hat = np.einsum('ik,jk->ijk',R_hat,R_hat)

        g_R = np.exp(1j*self.k_0*R)/(4*np.pi*R)
        expr_1 = (3/(self.k_0**2*R**2)-3j/(self.k_0*R)-1)*g_R
        expr_2 = (1+1j/(self.k_0*R)-1/(self.k_0**2*R**2))*g_R

        I = np.identity(3)
        G = (expr_1*RR_hat + expr_2*I.reshape(3,3,1)).T

        return G

    def data_aquisition(self):
        size = (len(self.dipoles),self.M_timepoints)
        phi = np.random.uniform(0,2*np.pi,size)
        theta = np.random.uniform(-np.pi/2,np.pi/2,size)

        polarizations = np.array([np.cos(phi)*np.sin(theta),
                                  np.sin(phi)*np.sin(theta),
                                  np.cos(theta)]).swapaxes(0,1)

        self.E_stack = np.zeros((3,self.N_sensors,self.M_timepoints),dtype=np.complex128)

        for i in range(len(self.dipoles)):
            G = self.dyadic_green(self.dipoles[i]).transpose(1,0,2)
            self.E_stack += G@polarizations[i]

    def noise_space(self):
        self.E_N = []
        for i in range(3):
            E_t = self.E_stack[i]
            S = E_t@np.conjugate(E_t).T

            eigvals,eigvecs = np.linalg.eig(S)

            dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)

            noice_idx = np.where(dist<1)[0]
            N = len(noice_idx)
            D = len(E_t)-N

            self.E_N.append(eigvecs[:,noice_idx])

    def check_if_resolvable(self):
        x1 = self.dipoles[0,0]
        x2 = self.dipoles[1,0]

        x = np.linspace(x1,x2,20).reshape(-1,1)
        y = self.dipoles[0,1]*np.ones_like(x)
        z = self.dipoles[0,2]*np.ones_like(x)
        pos = np.append(x,np.append(y,z,axis=1),axis=1)

        A = np.zeros((len(x),self.N_sensors,3,3),dtype=np.complex128)

        for i,dip in enumerate(pos):
            A[i] = self.dyadic_green(dip)

        P = np.zeros(len(x),dtype=np.complex128)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    P_1 = A[:,:,i,j].conj()@self.E_N[k]
                    P_2 = A[:,:,i,j]@self.E_N[k].conj()
                    P_t = (1/np.einsum('ij,ij->i',P_1,P_2))
                    P+= P_t.reshape(len(x))

        plt.plot(np.abs(P))
        plt.show()

        for el in P[P.argsort()[-2:]]:
            if el not in (P[0],P[-1]):
                return False

        peak = np.min((P[0],P[-1]))
        min = np.min(P)
        if min/peak <= 0.735:
            return True
        else:
            return False

    def find_resolution_limit(self):
        self.noise_space()
        self.old_dipoles = np.copy(self.dipoles)*2

        while 1:
            if self.check_if_resolvable():
                self.old_dipoles = np.copy(self.dipoles)
                self.E_stack_old = np.copy(self.E_stack)
                x1 = self.dipoles[0,0]
                x2 = self.dipoles[1,0]
                diff = np.abs(x1-x2)
                self.dipoles[0,0] = -diff/4
                self.dipoles[1,0] = diff/4
                self.resolution_limit = diff/self.wl
            else:
                if np.abs((self.dipoles[0,0]-self.old_dipoles[0,0])/self.old_dipoles[0,0]) < 0.01:
                    break
                x1 = self.dipoles[0,0]
                x2 = self.old_dipoles[0,0]
                x = np.abs(x1+x2)/2
                self.dipoles[0,0] = -x
                self.dipoles[1,0] = x

            self.data_aquisition()
            self.noise_space()

        self.dipoles = np.copy(self.old_dipoles)
        self.E_stack = np.copy(self.E_stack_old)

    def reconstruct_image(self,N_reconstruction):
        self.noise_space()

        x1 = self.dipoles[0,0]
        x2 = self.dipoles[1,0]
        FoV = np.abs(x1-x2)*1.5/self.wl

        x_pos = y_pos = np.linspace(-FoV/2,FoV/2,N_reconstruction)*self.wl
        z_pos = self.dipoles[0,2]

        xx,yy,zz = np.meshgrid(x_pos,y_pos,z_pos)
        grid = np.array((xx.flatten(),yy.flatten(),zz.flatten())).T

        A = np.zeros((len(grid),self.N_sensors,3,3),dtype=np.complex128)
        for i,el in enumerate(grid):
            A[i] = self.dyadic_green(el)

        P = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2]),dtype=np.complex128)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    P_1 = A[:,:,i,j].conj()@self.E_N[k]
                    P_2 = A[:,:,i,j]@self.E_N[k].conj()
                    P_t = (1/np.einsum('ij,ij->i',P_1,P_2))
                    P += P_t.reshape(xx.shape[0],xx.shape[1],xx.shape[2])

        self.P = P

    def save_image_stack(self,dir):
        t0 = round(time())

        data = {'Num_dipoles' : len(self.dipoles),
                'N_sensors' : self.E_stack.shape[0],
                'M_orientations' : self.E_stack.shape[1],
                'Dipole_positions' : self.dipoles.tolist()}

        with open(dir+"/{}.json".format(t0), 'w') as output:
            json.dump(data, output, indent=4)

        os.mkdir(dir+'/{}'.format(t0))

        for i in range(self.P.shape[2]):
            im = Image.fromarray(np.abs(self.P[:,:,i]).astype(np.float64))
            im.save(dir+'/{}'.format(t0)+'/{}.tiff'.format(i))

    def save_info(self,dir):
        t0 = round(time())

        try:
            data = {'Resolution limit [wl]' : self.resolution_limit,
                    'N_sensors' : self.N_sensors,
                    'Sensor radius' : self.radius,
                    'N_timepoints' : self.M_timepoints,
                    'Wavelength' : self.wl,
                    'n_objective' : self.n_obj,
                    'Relative permeability' : self.mur_obj,
                    'epsr_objective' : self.epsr_obj,
                    'NA' : self.NA,
                    'Theta max' : self.theta_max,
                    'Dipole_positions [wl]' : (self.dipoles/self.wl).tolist()}
        except:
            data = {'Resolution limit [wl]' : self.resolution_limit,
                    'N_sensors' : self.N_sensors,
                    'Sensor radius' : self.radius,
                    'N_timepoints' : self.M_timepoints,
                    'Wavelength' : self.wl,
                    'n_objective' : self.n_obj,
                    'Relative permeability' : self.mur_obj,
                    'epsr_objective' : self.epsr_obj,
                    'Theta max' : 0,
                    'Dipole_positions [wl]' : (self.dipoles/self.wl).tolist()}

        os.mkdir(dir+'/{}'.format(t0))

        with open(dir+"/{}/data.json".format(t0), 'w') as output:
            json.dump(data, output, indent=4)

        P = self.P.reshape(self.P.shape[0],self.P.shape[1])

        im = Image.fromarray(np.abs(P).astype(np.float64))
        im.save(dir+'/{}'.format(t0)+'/reconstruction.tiff')
