import numpy as np
from scipy.constants import epsilon_0, mu_0, c
from time import time
from scipy.integrate import quad_vec
from scipy.special import jv
import matplotlib.pyplot as plt
from numba import jit
import sys, os, json
from PIL import Image

def loadbar(counter,len):
    counter +=1
    done = (counter*100)//len+1
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % ('='*done, done))
    sys.stdout.flush()
    if counter == len:
        print('\n')

class Microscope:
    def __init__(self,Mag,N_sensors,wl,n,mur,epsr,k_0,f,NA,z_sub,dipoles,voxel_size,M_timepoints):
        self.Mag = Mag
        self.N_sensors = N_sensors
        self.voxel_size = voxel_size
        # self.cam_size = cam_size
        self.wl = wl
        self.n_obj, self.n_sub, self.n_cam = n
        self.mur_obj, self.mur_sub, self.mur_cam = mur
        self.epsr_obj, self.epsr_sub, self.epsr_cam = epsr
        self.k_0 = k_0
        self.f_obj, self.f_cam = f
        self.NA = NA
        self.z_sub = z_sub
        self.dipoles = dipoles
        self.M_timepoints = M_timepoints

        self.opt_ax_Mag = (self.n_cam/self.n_obj)*self.Mag**2
        self.theta_max = np.arcsin(self.NA/self.n_obj)
        self.k_obj = self.k_0*self.n_obj
        self.k_sub = self.k_0*self.n_sub
        self.k_cam = self.k_0*self.n_cam
        self.cam_size = np.sqrt(N_sensors)*voxel_size

        self.alpha = self.k_cam*np.exp(1j*(self.k_obj*self.f_obj+self.k_cam*self.f_cam))/(8j*np.pi)
        self.alpha *= self.f_obj/self.f_cam
        self.alpha *= np.sqrt(self.n_obj/self.n_cam)


    @staticmethod
    @jit(nopython=True, cache=True)
    def static_integrand(theta_obj,k,mur,f,cam_pos,dipole,z_sub):
        k_obj,k_sub,k_cam = k
        mur_obj,mur_sub = mur
        f_obj,f_cam = f
        x_cam,y_cam,z_cam = cam_pos
        x_dip,y_dip,z_dip = dipole

        kz_obj = k_obj*np.cos(theta_obj)

        kz_sub = np.sqrt(k_sub**2-(k_obj*np.sin(theta_obj))**2)

        RTE = (kz_obj/mur_obj-kz_sub/mur_sub) \
              /(kz_obj/mur_obj+kz_sub/mur_sub) \
              *np.exp(-2j*kz_obj*z_sub)

        Q = (kz_obj*k_sub**2*mur_obj) \
            /(kz_sub*k_obj**2*mur_sub)

        RTM = (Q-1)*np.exp(-2j*kz_obj*z_sub)/(Q+1)

        cosTheta_cam = np.sqrt(1-((f_obj/f_cam)**2)*(np.sin(theta_obj)**2))

        sinTheta_cam = f_obj*np.sin(theta_obj)/f_cam

        rho_x = k_cam*sinTheta_cam*x_cam \
                - k_obj*np.sin(theta_obj)*x_dip

        rho_y = k_cam*sinTheta_cam*y_cam \
                - k_obj*np.sin(theta_obj)*y_dip

        rho = np.sqrt(rho_y**2+rho_x**2)

        psi = np.arctan2(rho_y,rho_x)

        Zz = k_cam*cosTheta_cam*z_cam-k_obj*np.cos(theta_obj)*z_dip

        Z = k_cam*cosTheta_cam*z_cam+k_obj*np.cos(theta_obj)*z_dip

        fxx1 = (np.exp(1j*Zz)+RTE*np.exp(1j*Z)+cosTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM*np.exp(1j*Z))) \
               *np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

        fxx2 = (np.exp(1j*Zz)+RTE*np.exp(1j*Z)-cosTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM*np.exp(1j*Z))) \
               *np.cos(2*psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

        fxy = (np.exp(1j*Zz)+RTE*np.exp(1j*Z)-cosTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM*np.exp(1j*Z))) \
              *np.sin(2*psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

        fxz = -2j*cosTheta_cam*np.sin(theta_obj)*(np.exp(1j*Zz)+RTM*np.exp(1j*Z)) \
              *np.cos(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

        fyz = -2j*cosTheta_cam*np.sin(theta_obj)*(np.exp(1j*Zz)+RTM*np.exp(1j*Z)) \
              *np.sin(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

        fzx = 2j*sinTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM*np.exp(1j*Z)) \
              *np.cos(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

        fzy = 2j*sinTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM)*np.exp(1j*Z) \
              *np.sin(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

        fzz = -2*sinTheta_cam*np.sin(theta_obj)*(np.exp(1j*Zz)+RTM)*np.exp(1j*Z) \
              *np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

        f = (fxx1,fxx2,fxy,fxz,fyz,fzx,fzy,fzz,rho)

        return f

    def dyadic_green_microscope(self,cam_pos,dipole):
        def static(theta_obj):
            tmp = self.static_integrand(theta_obj,k,mur,f,cam_pos,dipole,self.z_sub)
            tmp,rho = tmp[:-1],tmp[-1]
            jvs = np.array([jv(0,rho),jv(2,rho),jv(2,rho),jv(1,rho),jv(1,rho),jv(1,rho),jv(1,rho),jv(0,rho)])
            return tmp*jvs

        k = np.array([self.k_obj,self.k_sub,self.k_cam])
        mur = np.array([self.mur_obj,self.mur_sub])
        f = np.array([self.f_obj,self.f_cam])

        integral = quad_vec(static, 0, self.theta_max)[0]

        Ixx1 = integral[0]
        Ixx2 = integral[1]
        Ixx = Ixx1+Ixx2
        Iyy = Ixx1-Ixx2
        Ixy = integral[2]
        Iyx = Ixy
        Ixz = integral[3]
        Iyz = integral[4]
        Izx = integral[5]
        Izy = integral[6]
        Izz = integral[7]

        G = self.alpha*np.array(((Ixx,Ixy,Ixz),(Iyx,Iyy,Iyz),(Izx,Izy,Izz)))
        return G

    def microscope_greens(self,points):
        grid_size = int(np.sqrt(self.N_sensors))
        x_cam = y_cam = np.linspace(-self.cam_size/2,self.cam_size/2,grid_size)*self.wl*self.Mag
        z_cam = 0


        G = np.zeros((len(points),len(x_cam),len(y_cam),3,3),dtype=np.complex128)
        tot_calc = len(y_cam)*len(x_cam)*len(points)
        for iy in range(len(y_cam)):
            y_pos = y_cam[iy]
            for ix in range(len(x_cam)):
                for id in range(len(points)):
                    x_pos = x_cam[ix]
                    pos_cam = np.array((x_pos,y_pos,z_cam))
                    G[id,ix,iy] = self.dyadic_green_microscope(pos_cam,points[id])

        return G

    def create_image_stack(self):
        G = self.microscope_greens(self.dipoles)

        phi = np.random.uniform(0,2*np.pi,(len(self.dipoles),self.M_timepoints))
        theta = np.random.uniform(-np.pi/2,np.pi/2,(len(self.dipoles),self.M_timepoints))

        polarizations = np.array([np.cos(phi)*np.sin(theta),
                                  np.sin(phi)*np.sin(theta),
                                  np.cos(theta)]).swapaxes(0,1)

        self.E_stack = np.zeros((3,self.N_sensors,self.M_timepoints),dtype=G.dtype)
        for i in range(len(self.dipoles)):
            G_t = G[i].transpose(2,0,1,3).reshape(3,-1,G[i].shape[-1])
            self.E_stack += G_t@polarizations[i]

    def find_noise_space(self):
        self.E_N = []
        for i in range(3):
            E_tmp = self.E_stack[i]
            S = E_tmp@np.conjugate(E_tmp).T

            eigvals,eigvecs = np.linalg.eig(S)

            dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)

            noice_idx = np.where(dist<1)[0]
            N = len(noice_idx)
            D = len(E_tmp)-N

            self.E_N.append(eigvecs[:,noice_idx])

    def check_if_resolvable(self):
        x1 = self.dipoles[0,0]
        x2 = self.dipoles[1,0]

        x = np.linspace(x1,x2,20).reshape(-1,1)
        y = self.dipoles[0,1]*np.ones_like(x)
        z = self.dipoles[0,2]*np.ones_like(x)
        pos = np.append(x,np.append(y,z,axis=1),axis=1)

        A = self.microscope_greens(pos)
        A = A.reshape(A.shape[0],-1,3,3)

        P = np.zeros(len(x),dtype=np.complex128)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    P_1 = A[:,:,i,j].conj()@self.E_N[k]
                    P_2 = A[:,:,i,j]@self.E_N[k].conj()
                    P_t = (1/np.einsum('ij,ij->i',P_1,P_2))
                    P+= P_t.reshape(len(x))

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
        self.find_noise_space()
        self.old_dipoles = np.copy(self.dipoles)*2

        counter = 0
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

            self.create_image_stack()
            self.find_noise_space()
            counter+=1
            print(counter)

        self.dipoles = np.copy(self.old_dipoles)
        self.E_stack = np.copy(self.E_stack_old)

    def reconstruct_image(self,FoV,N_reconstruction):
        self.find_noise_space()

        x_pos = y_pos = np.linspace(-FoV/2,FoV/2,N_reconstruction)*self.wl
        z_pos = self.dipoles[0,2]

        xx,yy,zz = np.meshgrid(x_pos,y_pos,z_pos)
        grid = np.array((xx.flatten(),yy.flatten(),zz.flatten())).T
        A = self.microscope_greens(grid)
        A = A.reshape(A.shape[0],-1,3,3)

        P = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2]),dtype=np.complex128)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    P_1 = A[:,:,i,j].conj()@self.E_N[k]
                    P_2 = A[:,:,i,j]@self.E_N[k].conj()
                    P_t = (1/np.einsum('ij,ij->i',P_1,P_2))
                    P += P_t.reshape(xx.shape[0],xx.shape[1],xx.shape[2])

        self.P = P

    def plot_fields(self):
        intensity_xpol = (np.abs(self.G[0,:,:,0,0]+self.G[1,:,:,0,0])**2)+(np.abs(self.G[0,:,:,1,0]+self.G[1,:,:,1,0])**2)+(np.abs(self.G[0,:,:,2,0]+self.G[1,:,:,2,0])**2)
        intensity_ypol = (np.abs(self.G[0,:,:,0,1]+self.G[1,:,:,0,1])**2)+(np.abs(self.G[0,:,:,1,1]+self.G[1,:,:,1,1])**2)+(np.abs(self.G[0,:,:,2,1]+self.G[1,:,:,2,1])**2)
        intensity_zpol = (np.abs(self.G[0,:,:,0,2]+self.G[1,:,:,0,2])**2)+(np.abs(self.G[0,:,:,1,2]+self.G[1,:,:,1,2])**2)+(np.abs(self.G[0,:,:,2,2]+self.G[1,:,:,2,2])**2)

        plt.subplot(3,3,1)
        plt.imshow(abs(self.G[0,:,:,0,0].T)**2)
        plt.colorbar()
        plt.subplot(3,3,2)
        plt.imshow(abs(self.G[0,:,:,1,0].T)**2)
        plt.colorbar()
        plt.subplot(3,3,3)
        plt.imshow(abs(self.G[0,:,:,2,0].T)**2)
        plt.colorbar()
        plt.subplot(3,3,4)
        plt.imshow(abs(self.G[1,:,:,0,0].T)**2)
        plt.colorbar()
        plt.subplot(3,3,5)
        plt.imshow(abs(self.G[1,:,:,1,0].T)**2)
        plt.colorbar()
        plt.subplot(3,3,6)
        plt.imshow(abs(self.G[1,:,:,2,0].T)**2)
        plt.colorbar()
        plt.subplot(3,3,8)
        plt.imshow(intensity_xpol.T)
        plt.colorbar()
        plt.show()

        plt.subplot(3,3,1)
        plt.imshow(abs(self.G[0,:,:,0,2].T)**2)
        plt.colorbar()
        plt.subplot(3,3,2)
        plt.imshow(abs(self.G[0,:,:,1,2].T)**2)
        plt.colorbar()
        plt.subplot(3,3,3)
        plt.imshow(abs(self.G[0,:,:,2,2].T)**2)
        plt.colorbar()
        plt.subplot(3,3,4)
        plt.imshow(abs(self.G[1,:,:,0,2].T)**2)
        plt.colorbar()
        plt.subplot(3,3,5)
        plt.imshow(abs(self.G[1,:,:,1,2].T)**2)
        plt.colorbar()
        plt.subplot(3,3,6)
        plt.imshow(abs(self.G[1,:,:,2,2].T)**2)
        plt.colorbar()
        plt.subplot(3,3,8)
        plt.imshow(intensity_zpol.T)
        plt.colorbar()
        plt.show()

        plt.subplot(2,2,1)
        plt.imshow(intensity_xpol.T)
        plt.colorbar()
        plt.subplot(2,2,2)
        plt.imshow(intensity_ypol.T)
        plt.colorbar()
        plt.subplot(2,2,3)
        plt.imshow(intensity_zpol.T)
        plt.colorbar()
        plt.show()

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

    def save_info(self,dir,counter):
        data = {'Resolution limit [wl]' : self.resolution_limit,
                'N_sensors' : str(self.N_sensors),
                'N_timepoints' : str(self.M_timepoints),
                'Magnification' : str(self.Mag),
                'Optical axis magnification' : self.opt_ax_Mag,
                'Camera FoV [wl]' : self.cam_size,
                'Wavelength' : self.wl,
                'n_objective' : self.n_obj,
                'n_substrate' : self.n_sub,
                'n_camera' : self.n_cam,
                'Relative permeability' : self.mur_obj,
                'epsr_objective' : self.epsr_obj,
                'epsr_substrate' : self.epsr_sub,
                'epsr_camera' : self.epsr_cam,
                'f_obj' : self.f_obj,
                'f_cam' : self.f_cam,
                'NA' : self.NA,
                'Theta max' : self.theta_max,
                'Substrate interface [m]' : self.z_sub,
                'Dipole_positions [wl]' : (self.dipoles/self.wl).tolist()}

        # os.mkdir(dir+'/{}_microscope'.format(counter))

        with open(dir+"/{}_data_microscope.json".format(counter), 'w') as output:
            json.dump(data, output, indent=4)

        # P = self.P.reshape(self.P.shape[0],self.P.shape[1])
        #
        # im = Image.fromarray(np.abs(P).astype(np.float64))
        # im.save(dir+'/{}'.format(t0)+'/reconstruction.tiff')
