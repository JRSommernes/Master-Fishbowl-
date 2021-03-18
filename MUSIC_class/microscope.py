import numpy as np
from scipy.constants import epsilon_0, mu_0, c
from time import time
from scipy.integrate import quad_vec
from scipy.special import jv
import matplotlib.pyplot as plt
from numba import jit
import sys

def loadbar(counter,len):
    counter +=1
    done = (counter*100)//len+1
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % ('='*done, done))
    sys.stdout.flush()
    if counter == len:
        print('\n')

class Fishbowl:
    def __init__(self,N_sensors,microscope_radius,n_0):
        self.N_sensors = N_sensors
        self.n_0 = n_0
        self.radius = microscope_radius

    def make_sensors(self):
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

class Microscope:
    def __init__(self,Mag,N_sensors,wl,n,mur,epsr,k_0,f,NA,z_sub,dipoles):
        self.Mag = Mag
        self.N_sensors = N_sensors
        self.wl = wl
        self.n_obj, self.n_sub, self.n_cam = n
        self.mur_obj, self.mur_sub, self.mur_cam = mur
        self.epsr_obj, self.epsr_sub, self.epsr_cam = epsr
        self.k_0 = k_0
        self.f_obj, self.f_cam = f
        self.NA = NA
        self.z_sub = z_sub
        self.dipoles = dipoles

        self.opt_ax_Mag = (self.n_cam/self.n_obj)*self.Mag**2
        self.theta_max = np.arcsin(self.NA/self.n_obj)
        self.k_obj = self.k_0*self.n_obj
        self.k_sub = self.k_0*self.n_sub
        self.k_cam = self.k_0*self.n_cam

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

    # def integrand(self,theta_obj):
    #
    #     x_cam,y_cam,z_cam = self.cam_pos
    #     x_dip,y_dip,z_dip = self.dipole
    #
    #     kz_obj = self.k_obj*np.cos(theta_obj)
    #
    #     kz_sub = np.sqrt(self.k_sub**2-(self.k_obj*np.sin(theta_obj))**2)
    #
    #     RTE = (kz_obj/self.mur_obj-kz_sub/self.mur_sub) \
    #           /(kz_obj/self.mur_obj+kz_sub/self.mur_sub) \
    #           *np.exp(-2j*kz_obj*self.z_sub)
    #
    #     Q = (kz_obj*self.k_sub**2*self.mur_obj) \
    #         /(kz_sub*self.k_obj**2*self.mur_sub)
    #
    #     RTM = (Q-1)*np.exp(-2j*kz_obj*self.z_sub)/(Q+1)
    #
    #     cosTheta_cam = np.sqrt(1-((self.f_obj/self.f_cam)**2)*(np.sin(theta_obj)**2))
    #
    #     sinTheta_cam = self.f_obj*np.sin(theta_obj)/self.f_cam
    #
    #     rho_x = self.k_cam*sinTheta_cam*x_cam \
    #             - self.k_obj*np.sin(theta_obj)*x_dip
    #
    #     rho_y = self.k_cam*sinTheta_cam*y_cam \
    #             - self.k_obj*np.sin(theta_obj)*y_dip
    #
    #     rho = np.sqrt(rho_y**2+rho_x**2)
    #
    #     psi = np.arctan2(rho_y,rho_x)
    #
    #     Zz = self.k_cam*cosTheta_cam*z_cam-self.k_obj*np.cos(theta_obj)*z_dip
    #
    #     Z = self.k_cam*cosTheta_cam*z_cam+self.k_obj*np.cos(theta_obj)*z_dip
    #
    #     fxx1 = (np.exp(1j*Zz)+RTE*np.exp(1j*Z)+cosTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM*np.exp(1j*Z))) \
    #            *jv(0,rho)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)
    #
    #     fxx2 = (np.exp(1j*Zz)+RTE*np.exp(1j*Z)-cosTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM*np.exp(1j*Z))) \
    #            *jv(2,rho)*np.cos(2*psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)
    #
    #     fxy = (np.exp(1j*Zz)+RTE*np.exp(1j*Z)-cosTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM*np.exp(1j*Z))) \
    #           *jv(2,rho)*np.sin(2*psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)
    #
    #     fxz = -2j*cosTheta_cam*np.sin(theta_obj)*(np.exp(1j*Zz)+RTM*np.exp(1j*Z)) \
    #           *jv(1,rho)*np.cos(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)
    #
    #     fyz = -2j*cosTheta_cam*np.sin(theta_obj)*(np.exp(1j*Zz)+RTM*np.exp(1j*Z)) \
    #           *jv(1,rho)*np.sin(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)
    #
    #     fzx = 2j*sinTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM*np.exp(1j*Z)) \
    #           *jv(1,rho)*np.cos(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)
    #
    #     fzy = 2j*sinTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM)*np.exp(1j*Z) \
    #           *jv(1,rho)*np.sin(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)
    #
    #     fzz = -2*sinTheta_cam*np.sin(theta_obj)*(np.exp(1j*Zz)+RTM)*np.exp(1j*Z) \
    #           *jv(0,rho)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)
    #
    #
    #     f = np.array((fxx1,fxx2,fxy,fxz,fyz,fzx,fzy,fzz))
    #
    #     return f

    def funGreenIntegrand1D(self,cam_pos,dipole):
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

    def microscope_greens(self,FoV,points):
        grid_size = int(np.sqrt(self.N_sensors))
        x_cam = y_cam = np.linspace(-FoV/2,FoV/2,grid_size)*self.wl*self.Mag
        z_cam = 0

        G = np.zeros((len(points),len(x_cam),len(y_cam),3,3),dtype=np.complex128)
        tot_calc = len(y_cam)*len(x_cam)*len(points)
        for iy in range(len(y_cam)):
            y_pos = y_cam[iy]
            for ix in range(len(x_cam)):
                for id in range(len(points)):
                    if id%100 == 0:
                        counter = id+ix*len(points)+iy*len(x_cam)*len(points)
                        loadbar(counter,tot_calc)
                    x_pos = x_cam[ix]
                    pos_cam = np.array((x_pos,y_pos,z_cam))
                    G[id,ix,iy] = self.funGreenIntegrand1D(pos_cam,points[id])

        return G

    def create_image_stack(self,FoV,stack_size):
        G = self.microscope_greens(FoV,self.dipoles)

        phi = np.random.uniform(0,2*np.pi,(len(self.dipoles),stack_size))
        theta = np.random.uniform(-np.pi/2,np.pi/2,(len(self.dipoles),stack_size))

        polarizations = np.array([np.cos(phi)*np.sin(theta),
                                  np.sin(phi)*np.sin(theta),
                                  np.cos(theta)]).swapaxes(0,1)

        self.E_stack = np.zeros((3*self.N_sensors,stack_size),dtype=G.dtype)
        for i in range(len(self.dipoles)):
            self.E_stack += G[i].reshape(-1,G[i].shape[-1])@polarizations[i,:]

    def find_noise_space(self):
        S = self.E_stack@np.conjugate(self.E_stack).T

        eigvals,eigvecs = np.linalg.eig(S)

        dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)

        noice_idx = np.where(dist<1)[0]
        N = len(noice_idx)
        D = len(self.E_stack)-N

        self.E_N = eigvecs[:,noice_idx]

    def reconstruct_image(self,FoV,N_reconstruction):
        self.find_noise_space()

        x_pos = y_pos = np.linspace(-FoV/2,FoV/2,N_reconstruction)*self.wl
        z_pos = 0

        xx,yy,zz = np.meshgrid(x_pos,y_pos,z_pos)
        grid = np.array((xx.flatten(),yy.flatten(),zz.flatten())).T
        A = self.microscope_greens(FoV,grid)
        A = A.reshape(A.shape[0],-1,A.shape[-1])

        try:
            P = np.zeros((3,len(x_pos),len(y_pos),len(z_pos)),dtype=np.complex128)
        except:
            P = np.zeros((3,len(x_pos),len(y_pos)),dtype=np.complex128)
        for i in range(3):
            A_t = A[:,:,i]

            P_1 = np.matmul(np.conjugate(A_t),self.E_N)
            P_2 = np.matmul(A_t,np.conjugate(self.E_N))

            P_t = (1/np.einsum('ij,ij->i',P_1,P_2))
            try:
                P[i] = P_t.reshape((len(x_pos),len((y_pos),len(z_pos))))
            except:
                P[i] = P_t.reshape((len(x_pos),len((y_pos))))

        P = np.sum(P,axis=0)

        plt.imshow(np.abs(P))
        plt.show()

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
