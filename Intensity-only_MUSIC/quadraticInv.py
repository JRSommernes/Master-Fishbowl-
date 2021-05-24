import numpy as np
import matplotlib.pyplot as plt
from phaseLift import generateSimulatedField
from misc_functions import dyadic_green, loadbar

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
    maxIte = int(1e4)
    # maxIte = 100
    N = mat_am.shape[1]
    randVal = np.random.uniform(-30,30,size=(N,2))
    # randVal = np.zeros((N,2))
    # randVal = np.ones_like(randVal)
    i_x = (randVal[:,0] + 1j*randVal[:,1]).reshape(-1,1)
    thre_absErr = 1e-3
    thre_relErr = 1e-5
    thre_relErr_xEst = 1e-5

    vec_absErr = np.zeros((maxIte,1))
    vec_relErr = np.ones((maxIte,1))
    vec_relErr_xEst = np.ones((maxIte,1))
    recSol_xEst = np.zeros((N,maxIte),dtype=np.complex128)

    for iter in range(maxIte):
        if iter%10 == 0:
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

        if (vec_absErr[iter] < thre_absErr) or (vec_relErr[iter] < thre_relErr) or (vec_relErr_xEst[iter] < thre_relErr_xEst):
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

if __name__ == '__main__':
    mat_am, b, xTrue = generateSimulatedField()

    print(mat_am.shape,b.shape,xTrue.shape)
    print(mat_am.dtype,b.dtype,xTrue.dtype)
    exit()


    xEst = algQuadraticInv(mat_am,b)

    normalized_xTrue = xTrue*np.conj(xTrue[0])/np.abs(np.conj(xTrue[0]))
    normalized_xEst = xEst*np.conj(xEst[0])/np.abs(np.conj(xEst[0]))
    N0 = (len(xEst)-1)/2
    vecn = np.arange(-N0,N0+1)

    plt.plot(vecn,np.angle(normalized_xTrue),'o')
    plt.plot(vecn,np.angle(normalized_xEst),'*')
    plt.xlim([-12, 12])
    plt.show()

    plt.plot(vecn,np.abs(normalized_xTrue),'o')
    plt.plot(vecn,np.abs(normalized_xEst),'*')
    plt.xlim([-12, 12])
    plt.show()
