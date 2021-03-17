import numpy as np
from scipy.special import jv
from scipy.integrate import quad
import matplotlib.pyplot as plt

def funGreenIntegrand1D(pos_cam,dipole,things):
    x_cam = pos_cam[0]
    y_cam = pos_cam[1]
    z_cam = pos_cam[2]

    x_dip = dipole[0]
    y_dip = dipole[1]
    z_dip = dipole[2]

    k_cam,k_obj,f_cam,f_obj,n_cam,n_obj,k_sub,theta_max,z_Interface_sub,\
    mur_obj,mur_sub = things

    alpha = k_cam*np.exp(1j*(k_obj*f_obj+k_cam*f_cam))/(8j*np.pi)
    alpha *= f_obj/f_cam
    alpha *= np.sqrt(n_obj/n_cam)

    kz_obj = lambda theta_obj: k_obj*np.cos(theta_obj)
    kz_sub = lambda theta_obj: np.sqrt(k_sub**2-(k_obj*np.sin(theta_obj))**2)
    RTE = lambda theta_obj: (kz_obj(theta_obj)/mur_obj-kz_sub(theta_obj)/mur_sub)/(kz_obj(theta_obj)/mur_obj+kz_sub(theta_obj)/mur_sub)*np.exp(-2j*kz_obj(theta_obj)*z_Interface_sub)
    Q = lambda theta_obj: (kz_obj(theta_obj)*k_sub**2*mur_obj)/(kz_sub(theta_obj)*k_obj**2*mur_sub)
    RTM = lambda theta_obj: (Q(theta_obj)-1)*np.exp(-2j*kz_obj(theta_obj)*z_Interface_sub)/(Q(theta_obj)+1)
    cosTheta_cam = lambda theta_obj: np.sqrt(1-((f_obj/f_cam)**2)*(np.sin(theta_obj)**2))
    sinTheta_cam = lambda theta_obj: f_obj*np.sin(theta_obj)/f_cam
    rho_x = lambda theta_obj: k_cam*sinTheta_cam(theta_obj)*x_cam \
                              - k_obj*np.sin(theta_obj)*x_dip
    rho_y = lambda theta_obj: k_cam*sinTheta_cam(theta_obj)*y_cam \
                              - k_obj*np.sin(theta_obj)*y_dip
    rho = lambda theta_obj: np.sqrt(rho_y(theta_obj)**2+rho_x(theta_obj)**2)
    psi = lambda theta_obj: np.arctan2(rho_y(theta_obj),rho_x(theta_obj))
    Zz = lambda theta_obj: k_cam*cosTheta_cam(theta_obj)*z_cam-k_obj*np.cos(theta_obj)*z_dip
    Z = lambda theta_obj: k_cam*cosTheta_cam(theta_obj)*z_cam+k_obj*np.cos(theta_obj)*z_dip

    fxx1 = lambda theta_obj: (np.exp(1j*Zz(theta_obj))+RTE(theta_obj)*np.exp(1j*Z(theta_obj))+cosTheta_cam(theta_obj)*np.cos(theta_obj)*(np.exp(1j*Zz(theta_obj))-RTM(theta_obj)*np.exp(1j*Z(theta_obj)))) \
                             * jv(0,rho(theta_obj))*np.sqrt(np.cos(theta_obj)/cosTheta_cam(theta_obj))*np.sin(theta_obj)
    fxx1_real = lambda theta_obj: np.real(fxx1(theta_obj))
    fxx1_imag = lambda theta_obj: np.imag(fxx1(theta_obj))
    Ixx1_real = quad(fxx1_real,0,theta_max)[0]
    Ixx1_imag = quad(fxx1_imag,0,theta_max)[0]
    Ixx1 = Ixx1_real + 1j*Ixx1_imag

    fxx2 = lambda theta_obj: (np.exp(1j*Zz(theta_obj))+RTE(theta_obj)*np.exp(1j*Z(theta_obj))-cosTheta_cam(theta_obj)*np.cos(theta_obj)*(np.exp(1j*Zz(theta_obj))-RTM(theta_obj)*np.exp(1j*Z(theta_obj)))) \
                             * jv(2,rho(theta_obj))*np.cos(2*psi(theta_obj))*np.sqrt(np.cos(theta_obj)/cosTheta_cam(theta_obj))*np.sin(theta_obj)
    fxx2_real = lambda theta_obj: np.real(fxx2(theta_obj))
    fxx2_imag = lambda theta_obj: np.imag(fxx2(theta_obj))
    Ixx2_real = quad(fxx2_real,0,theta_max)[0]
    Ixx2_imag = quad(fxx2_imag,0,theta_max)[0]
    Ixx2 = Ixx2_real + 1j*Ixx2_imag

    Ixx = Ixx1+Ixx2
    Iyy = Ixx1-Ixx2

    fxy = lambda theta_obj: (np.exp(1j*Zz(theta_obj))+RTE(theta_obj)*np.exp(1j*Z(theta_obj))-cosTheta_cam(theta_obj)*np.cos(theta_obj)*(np.exp(1j*Zz(theta_obj))-RTM(theta_obj)*np.exp(1j*Z(theta_obj)))) \
          * jv(2,rho(theta_obj))*np.sin(2*psi(theta_obj))*np.sqrt(np.cos(theta_obj)/cosTheta_cam(theta_obj))*np.sin(theta_obj)
    fxy_real = lambda theta_obj: np.real(fxy(theta_obj))
    fxy_imag = lambda theta_obj: np.imag(fxy(theta_obj))
    Ixy_real = quad(fxy_real,0,theta_max)[0]
    Ixy_imag = quad(fxy_imag,0,theta_max)[0]
    Ixy = Ixy_real + 1j*Ixy_imag
    Iyx = Ixy

    fxz = lambda theta_obj: -2j*cosTheta_cam(theta_obj)*np.sin(theta_obj)*(np.exp(1j*Zz(theta_obj))+RTM(theta_obj)*np.exp(1j*Z(theta_obj))) \
                            *jv(1,rho(theta_obj))*np.cos(psi(theta_obj))*np.sqrt(np.cos(theta_obj)/cosTheta_cam(theta_obj))*np.sin(theta_obj)
    fxz_real = lambda theta_obj: np.real(fxz(theta_obj))
    fxz_imag = lambda theta_obj: np.imag(fxz(theta_obj))
    Ixz_real = quad(fxz_real,0,theta_max)[0]
    Ixz_imag = quad(fxz_imag,0,theta_max)[0]
    Ixz = Ixz_real + 1j*Ixz_imag

    fyz = lambda theta_obj: -2j*cosTheta_cam(theta_obj)*np.sin(theta_obj)*(np.exp(1j*Zz(theta_obj))+RTM(theta_obj)*np.exp(1j*Z(theta_obj))) \
                            *jv(1,rho(theta_obj))*np.sin(psi(theta_obj))*np.sqrt(np.cos(theta_obj)/cosTheta_cam(theta_obj))*np.sin(theta_obj)
    fyz_real = lambda theta_obj: np.real(fyz(theta_obj))
    fyz_imag = lambda theta_obj: np.imag(fyz(theta_obj))
    Iyz_real = quad(fyz_real,0,theta_max)[0]
    Iyz_imag = quad(fyz_imag,0,theta_max)[0]
    Iyz = Iyz_real + 1j*Iyz_imag

    fzx = lambda theta_obj: 2j*sinTheta_cam(theta_obj)*np.cos(theta_obj)*(np.exp(1j*Zz(theta_obj))-RTM(theta_obj)*np.exp(1j*Z(theta_obj))) \
                            *jv(1,rho(theta_obj))*np.cos(psi(theta_obj))*np.sqrt(np.cos(theta_obj)/cosTheta_cam(theta_obj))*np.sin(theta_obj)
    fzx_real = lambda theta_obj: np.real(fzx(theta_obj))
    fzx_imag = lambda theta_obj: np.imag(fzx(theta_obj))
    Izx_real = quad(fzx_real,0,theta_max)[0]
    Izx_imag = quad(fzx_imag,0,theta_max)[0]
    Izx = Izx_real + 1j*Izx_imag

    fzy = lambda theta_obj: 2j*sinTheta_cam(theta_obj)*np.cos(theta_obj)*(np.exp(1j*Zz(theta_obj))-RTM(theta_obj))*np.exp(1j*Z(theta_obj)) \
                            *jv(1,rho(theta_obj))*np.sin(psi(theta_obj))*np.sqrt(np.cos(theta_obj)/cosTheta_cam(theta_obj))*np.sin(theta_obj)
    fzy_real = lambda theta_obj: np.real(fzy(theta_obj))
    fzy_imag = lambda theta_obj: np.imag(fzy(theta_obj))
    Izy_real = quad(fzy_real,0,theta_max)[0]
    Izy_imag = quad(fzy_imag,0,theta_max)[0]
    Izy = Izy_real + 1j*Izy_imag

    fzz = lambda theta_obj: -2*sinTheta_cam(theta_obj)*np.sin(theta_obj)*(np.exp(1j*Zz(theta_obj))+RTM(theta_obj))*np.exp(1j*Z(theta_obj)) \
                            *jv(0,rho(theta_obj))*np.sqrt(np.cos(theta_obj)/cosTheta_cam(theta_obj))*np.sin(theta_obj)
    fzz_real = lambda theta_obj: np.real(fzz(theta_obj))
    fzz_imag = lambda theta_obj: np.imag(fzz(theta_obj))
    Izz_real = quad(fzz_real,0,theta_max)[0]
    Izz_imag = quad(fzz_imag,0,theta_max)[0]
    Izz = Izz_real + 1j*Izz_imag

    solG = alpha*np.array(((Ixx,Ixy,Ixz),(Iyx,Iyy,Iyz),(Izx,Izy,Izz)))
    return solG



def microscope_greens(dipoles,wl,M_inputs,N_sensors,k_0,Mag):
    N_cam = int(np.sqrt(N_sensors))
    z_cam = 0
    x_cam = y_cam = np.linspace(-3,3,N_cam)*wl*Mag

    n_obj = 1.33
    mur_obj = 1
    epsr_obj = n_obj**2/mur_obj
    k_obj = k_0*n_obj

    mur_sub = 1
    n_sub = 4.3
    epsr_sub = n_sub**2/mur_sub
    k_sub = k_0*n_sub
    z_Interface_sub = -30e-9

    mur_cam = 1
    n_cam = 1
    epsr_cam = n_cam**2/mur_cam
    k_cam = k_0*n_cam

    f_cam = 16e-2
    f_obj = f_cam/(Mag*n_cam/n_obj)
    NA = 1.2
    theta_max = np.arcsin(NA/n_obj)

    opt_ax_Mag = (n_cam/n_obj)*Mag**2

    things = [k_cam,k_obj,f_cam,f_obj,n_cam,n_obj,k_sub,theta_max,\
              z_Interface_sub,mur_obj,mur_sub]

    solG = np.zeros((2,len(x_cam),len(y_cam),3,3),dtype=np.complex128)

    for iy in range(len(y_cam)):
        print(iy)
        y_pos = y_cam[iy]
        for ix in range(len(x_cam)):
            x_pos = x_cam[ix]
            pos_cam = np.array((x_pos,y_pos,z_cam))
            solG[0,ix,iy] = funGreenIntegrand1D(pos_cam,dipoles[0],things)
            solG[1,ix,iy] = funGreenIntegrand1D(pos_cam,dipoles[1],things)

    intensity_xpol = (np.abs(solG[0,:,:,0,0]+solG[1,:,:,0,0])**2)+(np.abs(solG[0,:,:,1,0]+solG[0,:,:,1,0])**2)+(np.abs(solG[0,:,:,2,0]+solG[1,:,:,2,0])**2)
    intensity_zpol = (np.abs(solG[0,:,:,0,2]+solG[1,:,:,0,2])**2)+(np.abs(solG[0,:,:,1,2]+solG[0,:,:,1,2])**2)+(np.abs(solG[0,:,:,2,2]+solG[1,:,:,2,2])**2)

    plt.subplot(3,3,1)
    plt.imshow(abs(solG[0,:,:,0,0])**2)
    plt.subplot(3,3,2)
    plt.imshow(abs(solG[0,:,:,1,0])**2)
    plt.subplot(3,3,3)
    plt.imshow(abs(solG[0,:,:,2,0])**2)
    plt.subplot(3,3,4)
    plt.imshow(abs(solG[1,:,:,0,0])**2)
    plt.subplot(3,3,5)
    plt.imshow(abs(solG[1,:,:,1,0])**2)
    plt.subplot(3,3,6)
    plt.imshow(abs(solG[1,:,:,2,0])**2)
    plt.subplot(3,3,8)
    plt.imshow(intensity_xpol)
    plt.show()


    plt.subplot(3,3,1)
    plt.imshow(abs(solG[0,:,:,0,2])**2)
    plt.subplot(3,3,2)
    plt.imshow(abs(solG[0,:,:,1,2])**2)
    plt.subplot(3,3,3)
    plt.imshow(abs(solG[0,:,:,2,2])**2)
    plt.subplot(3,3,4)
    plt.imshow(abs(solG[1,:,:,0,2])**2)
    plt.subplot(3,3,5)
    plt.imshow(abs(solG[1,:,:,1,2])**2)
    plt.subplot(3,3,6)
    plt.imshow(abs(solG[1,:,:,2,2])**2)
    plt.subplot(3,3,8)
    plt.imshow(intensity_zpol)
    plt.show()
