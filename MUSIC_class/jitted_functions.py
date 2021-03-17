import numpy as np
from numba import jit
from scipy.special import jv

# @jit(nopython=True, cache=True)
def something(theta_obj,k,mur,f,cam_pos,dipole,z_Interface_sub):
    k_obj,k_sub,k_cam = k
    mur_obj,mur_sub = mur
    f_obj,f_cam = f
    x_cam,y_cam,z_cam = cam_pos
    x_dip,y_dip,z_dip = dipole

    kz_obj = k_obj*np.cos(theta_obj)

    kz_sub = np.sqrt(k_sub**2-(k_obj*np.sin(theta_obj))**2)

    RTE = (kz_obj/mur_obj-kz_sub/mur_sub) \
          /(kz_obj/mur_obj+kz_sub/mur_sub) \
          *np.exp(-2j*kz_obj*z_Interface_sub)

    Q = (kz_obj*k_sub**2*mur_obj) \
        /(kz_sub*k_obj**2*mur_sub)

    RTM = (Q-1)*np.exp(-2j*kz_obj*z_Interface_sub)/(Q+1)

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
           *jv(0,rho)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

    fxx2 = (np.exp(1j*Zz)+RTE*np.exp(1j*Z)-cosTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM*np.exp(1j*Z))) \
           *jv(2,rho)*np.cos(2*psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

    fxy = (np.exp(1j*Zz)+RTE*np.exp(1j*Z)-cosTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM*np.exp(1j*Z))) \
          *jv(2,rho)*np.sin(2*psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

    fxz = -2j*cosTheta_cam*np.sin(theta_obj)*(np.exp(1j*Zz)+RTM*np.exp(1j*Z)) \
          *jv(1,rho)*np.cos(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

    fyz = -2j*cosTheta_cam*np.sin(theta_obj)*(np.exp(1j*Zz)+RTM*np.exp(1j*Z)) \
          *jv(1,rho)*np.sin(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

    fzx = 2j*sinTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM*np.exp(1j*Z)) \
          *jv(1,rho)*np.cos(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

    fzy = 2j*sinTheta_cam*np.cos(theta_obj)*(np.exp(1j*Zz)-RTM)*np.exp(1j*Z) \
          *jv(1,rho)*np.sin(psi)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

    fzz = -2*sinTheta_cam*np.sin(theta_obj)*(np.exp(1j*Zz)+RTM)*np.exp(1j*Z) \
          *jv(0,rho)*np.sqrt(np.cos(theta_obj)/cosTheta_cam)*np.sin(theta_obj)

    f = np.array((fxx1,fxx2,fxy,fxz,fyz,fzx,fzy,fzz))

    return f
