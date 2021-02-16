import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from misc_functions import dyadic_green, loadbar
from time import time
from phaseLift import *

Nfeval = 1

def compGradient(x,mat_am,b):
    vecE = np.conj(mat_am)@x
    b_est = vecE*np.conj(vecE)
    g = (mat_am.T)@np.diag(b_est.flatten()-b.flatten())@np.conj(mat_am)@x
    g = g/mat_am.shape[0]
    return g

def compuStepSize(x,g,mat_am,b):
    adx = np.conj(mat_am)@g
    c3 = np.sum(np.abs(adx)**4)

    ax = np.conj(np.conj(mat_am)@x)
    Re_ax = (ax*adx).real
    c2 = -3*np.sum((np.abs(adx)**2)*Re_ax)

    c1 = np.sum(2*Re_ax**2 + (np.abs(adx)**2)*((np.abs(ax)**2)-b))

    c0 = -np.sum(Re_ax*((np.abs(ax)**2)-b))

    solt = np.roots((c3,c2,c1,c0))
    idRealVal = np.where(np.abs(solt.imag)<1e-20)[0]
    if len(idRealVal) == 3:
        dataFitErr = np.zeros((3,1))
        for it in range(0,3):
            xnew = x - solt[it]*g
            ax = np.conj(np.conj(mat_am)@xnew)
            dataFitErr[it] = np.linalg.norm((np.abs(ax)**2)-b)

        idMin = np.argmin(dataFitErr)
        t = solt[idMin]
    elif len(idRealVal) == 1:
        t = solt[idRealVal]

    return t

def algQuadraticInv(mat_am,b):
    maxIte = int(1e6)
    # maxIte = 100
    N = mat_am.shape[1]
    randVal = np.random.uniform(-1,1,size=(N,2))
    # i_x = (randVal[:,0] + 1j*randVal[:,1]).reshape(-1,1)
    i_x = randVal[:,0].reshape(-1,1)
    thre_absErr = 1e-3
    thre_relErr = 1e-5
    thre_relErr_xEst = 1e-5

    vec_absErr = np.zeros((maxIte,1))
    vec_relErr = np.ones((maxIte,1))
    vec_relErr_xEst = np.ones((maxIte,1))
    recSol_xEst = np.zeros((N,maxIte),dtype=np.complex128)

    t0 = time()
    big_number = 3000
    for iter in range(maxIte):
        if iter%100 == 0:
            loadbar(iter,maxIte)
        g = compGradient(i_x,mat_am,b)
        t = compuStepSize(i_x,g,mat_am,b)
        i_x = i_x - t*g

        recSol_xEst[:,iter] = i_x.flatten()
        ax = np.conj(np.conj(mat_am)@i_x)
        vec_absErr[iter] = np.linalg.norm(b-(np.abs(ax)**2))**2

        if iter > 0:
            vec_relErr[iter] = np.abs(vec_absErr[iter]-vec_absErr[iter-1])/vec_absErr[iter]
            vec_relErr_xEst[iter] = np.max(np.abs(recSol_xEst[:,iter]-recSol_xEst[:,iter-1])/np.abs(recSol_xEst[:,iter]))

        if (vec_absErr[iter] < thre_absErr):
            vec_absErr = vec_absErr[:(iter+1)]
            vec_relErr = vec_relErr[:(iter+1)]
            vec_relErr_xEst = vec_relErr_xEst[:(iter+1)]
            recSol_xEst = recSol_xEst[:,:(iter+1)]
            break

        if time()-t0 > big_number:
            vec_absErr = vec_absErr[:(iter+1)]
            vec_relErr = vec_relErr[:(iter+1)]
            vec_relErr_xEst = vec_relErr_xEst[:(iter+1)]
            recSol_xEst = recSol_xEst[:,:(iter+1)]
            break

    xEst = i_x

    plt.semilogy(np.arange(0,len(vec_absErr)),vec_absErr)
    plt.show()

    plt.semilogy(np.arange(0,len(vec_relErr)),vec_relErr)
    plt.show()

    plt.semilogy(np.arange(0,len(vec_relErr_xEst)),vec_relErr_xEst)
    plt.show()

    return xEst


def objective(x,Am,b):
    return np.sum(np.abs(Am.conjugate()@x)**2-b)

def callbackF(Xi,Am,b):
    global Nfeval
    print('{0:4d}   {1: 3.6f}'.format(Nfeval, objective(Xi,Am,b)))
    Nfeval += 1

def test_min(mat_am,b):
    maxIte = int(1e4)
    N = mat_am.shape[1]
    i_x = np.random.uniform(-1,1,size=(N))
    b = b.flatten()

    bond = (-1.0,1.0)
    bnds = ((bond,)*N)

    sol = minimize(objective, i_x, args=(mat_am,b), callback=callbackF(i_x,mat_am,b))









def plot_sensor_field(sensors,E):
    x = sensors[0]
    y = sensors[1]

    plt.scatter(x,y,c=np.abs(E))
    plt.colorbar()
    plt.show()

def make_2d_sensor_grid(M0,N,wl):
    phi = np.linspace(0,2*np.pi,M0)
    rho = np.arange(5,5+2*N,2)*wl

    phi_mat, rho_mat = np.meshgrid(phi,rho)

    xx = rho_mat*np.cos(phi_mat)
    x = xx.flatten()
    yy = rho_mat*np.sin(phi_mat)
    y = yy.flatten()
    z = np.zeros_like(y)

    return np.array([x,y,z])

def make_dipoles():
    dipole_pos = np.array([[-0.5*wl,0,0],[0.5*wl,0,0]])
    # dipole_pos = np.array([[0,0,0]])


    phi = np.random.uniform(0,2*np.pi,(len(dipole_pos)))
    theta = np.random.uniform(-np.pi/2,np.pi/2,(len(dipole_pos)))

    # phi = np.ones_like(phi)*0
    # theta = np.ones_like(phi)*np.pi


    polarizations = np.array([np.cos(phi)*np.sin(theta),
                              np.sin(phi)*np.sin(theta),
                              np.cos(theta)]).swapaxes(0,1)

    return dipole_pos, polarizations

def find_xTrue(FoV,M,pos,pol):
    xTrue = np.zeros((M,M,3))

    x = np.linspace(-FoV/2,FoV/2,M)
    y = np.linspace(-FoV/2,FoV/2,M)
    for i,p in enumerate(pos):
        x_idx = np.argmin(np.abs(x-p[0]))
        y_idx = np.argmin(np.abs(y-p[1]))
        xTrue[x_idx,y_idx] = pol[i]

    return xTrue

def make_Am(sensors,wl,M,FoV):
    N = sensors.shape[1]

    x = np.linspace(-FoV/2,FoV/2,M)
    y = np.linspace(-FoV/2,FoV/2,M)

    G = np.zeros((M,M,N,3,3),dtype=np.complex128)
    for i,xx in enumerate(x):
        for j,yy in enumerate(y):
            pos = np.array([xx,yy,0])
            G[i,j] += dyadic_green(sensors,pos,k_0)

    # G_th = G[:,:,:,2].transpose(3,0,1,2)
    #
    # phi = np.angle(sensors[0]+sensors[1]*1j)
    # phi = phi*(phi>0)+(2*np.pi+phi)*(phi<=0)
    # phi = (phi*180/np.pi).astype(int)
    # phi = phi%90
    # alpha = (90-phi)*np.pi/180
    # G_phi_1 = G[:,:,:,0].transpose(3,0,1,2)*np.cos(alpha)
    # G_phi_2 = G[:,:,:,1].transpose(3,0,1,2)*np.cos(phi)
    # G_ph = G_phi_1 + G_phi_2
    #
    # return G_th,G_ph

    G = G.transpose(3,0,1,2,4)
    G = G.reshape((*G.shape[:3],-1))
    return G


def make_E_field(sensors,wl,k_0):
    N = sensors.shape[1]
    pos, pol = make_dipoles()
    FoV = 2*wl
    NN = 5

    G = np.zeros((N,3),dtype=np.complex128)
    for i in range(len(pos)):
        G += dyadic_green(sensors,pos[i],k_0)@pol[i]

    # E_theta = G[:,2]
    # # E_phi = G[:,0]+G[:,1]
    #
    # phi = np.angle(sensors[0]+sensors[1]*1j)
    # phi = phi*(phi>0)+(2*np.pi+phi)*(phi<=0)
    # phi = (phi*180/np.pi).astype(int)
    # phi = phi%90
    # alpha = (90-phi)*np.pi/180
    # E_phi_1 = G[:,0]*np.cos(alpha)
    # E_phi_2 = G[:,1]*np.cos(phi)
    # E_phi = E_phi_1 + E_phi_2
    #
    # I_th = np.abs(E_theta)**2
    # I_ph = np.abs(E_phi)**2

    E = G.flatten()
    b = np.abs(E)**2

    Am = make_Am(sensors,wl,NN,FoV)

    # Am = np.append(Am_th, Am_ph, axis=-1)



    # b = np.append(I_th, I_ph)
    # E_true = np.append(E_theta, E_phi)

    xTrue = find_xTrue(FoV,NN,pos,pol).transpose(2,0,1)


    # test1 = Am.reshape(-1, Am.shape[-1])
    # test2 = xTrue.flatten().reshape(-1,1)
    #
    # test3 = test1.transpose()@test2
    # test4 = np.abs(test3)**2
    #
    # print(test1.shape,test2.shape)
    # print(np.unique(np.equal(test3,b)))
    #
    # plt.plot(test3.real,c='b')
    # plt.plot(E_true.real,c='r')
    # plt.show()
    # plt.plot(test3.imag,c='b')
    # plt.plot(E_true.imag,c='r')
    # plt.show()
    # exit()

    Am = Am.reshape(-1, Am.shape[-1]).T
    b = b.reshape(-1, 1)
    xTrue = xTrue.flatten()

    return Am, b, E, xTrue


if __name__ == '__main__':
    eps_0 = 8.8541878128e-12
    mu_0 = 4*np.pi*10**-7
    c_0 = 1/np.sqrt(eps_0*mu_0)

    wl = 690e-9
    freq = c_0/wl
    k_0 = 2*np.pi*freq*np.sqrt(eps_0*mu_0)

    M0 = 60
    N = 4

    sensors = make_2d_sensor_grid(M0,N,wl)

    Am, b, E_true, xTrue = make_E_field(sensors,wl,k_0)

    xEst = algQuadraticInv(Am,b)
    # xEst = test_min(Am,b)



    # normalized_xTrue = xTrue*np.conj(xTrue[0])/np.abs(np.conj(xTrue[0]))
    # normalized_xEst = xEst*np.conj(xEst[0])/np.abs(np.conj(xEst[0]))
    # N0 = (len(xEst)-1)/2
    # vecn = np.arange(-N0,N0+1)

    # plt.plot(vecn,np.angle(normalized_xTrue))
    # plt.plot(vecn,np.angle(normalized_xEst),'*')
    # plt.xlim([-12, 12])
    plt.plot(xTrue)
    plt.plot(xEst.real)
    plt.show()

    # plt.plot(vecn,np.abs(normalized_xTrue))
    # plt.plot(vecn,np.abs(normalized_xEst),'*')
    # plt.xlim([-12, 12])
    # plt.show()
