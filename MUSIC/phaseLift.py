import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2
from misc_functions import loadbar

def generateSimulatedField():
    N0 = 12
    lambda0 = 1e-3
    beta = N0/(1.9*lambda0)
    vecn = np.arange(-N0,N0+1,1)
    xTrue = 10*np.exp(1j*vecn)
    N = 2*N0+1
    # Position of observation points
    rho1 = 3*lambda0
    rho2 = 5*lambda0
    rho3 = 7*lambda0
    rho4 = 9*lambda0
    vec_rho = np.array([rho1,rho2,rho3,rho4])
    M0 = 49
    vec_phi = np.linspace(2*np.pi/M0,2*np.pi,M0)
    mat_rho,mat_phi = np.meshgrid(vec_rho,vec_phi)
    M = mat_rho.size
    # Electric field at observation points
    rho_obs = np.repeat(mat_rho.T.flatten().reshape(M,1),N,axis=1)
    phi_obs = np.repeat(mat_phi.T.flatten().reshape(M,1),N,axis=1)
    mat_n = np.repeat(vecn.reshape(1,N),M,axis=0)
    mat_am = np.conj(hankel2(mat_n,beta*rho_obs)*np.exp(1j*mat_n*phi_obs))
    vecE = np.conj(mat_am)@xTrue.reshape(-1,1)

    b = vecE*np.conj(vecE)
    b = b.real

    return mat_am, b, xTrue

def compoMatrix4Inv(mat_am,xTrue):
    M,N = mat_am.shape

    AL = np.zeros((M,N**2),dtype=np.complex128)
    for m in range(M):
        Am = mat_am[m].reshape(N,1) @ np.conj(mat_am[m].reshape(1,N))
        AmT = Am.T
        AL[m] = Am.flatten()

    XTrue = xTrue.reshape(-1,1)@np.conj(xTrue).reshape(1,-1)

    nn = np.arange(1,N)
    NN = np.sum(nn)
    F = np.zeros((M,NN),dtype=np.complex128)
    XT1 = np.zeros(NN,dtype=np.complex128)
    vecXTrue = XTrue.T.flatten()
    cnt = 0
    for n in range(2,N+1):
        idCol = np.arange(n,N+1)+(n-2)*N-1
        tmp = cnt+len(idCol)
        F[:,cnt:tmp] = AL[:,idCol]
        XT1[cnt:tmp] = vecXTrue[idCol]
        cnt = tmp

    H = np.zeros((M,N),dtype=np.complex128)
    XD = np.zeros(N,dtype=np.complex128)
    for n in range(1,N+1):
        idCol = (n-1)*(N+1)
        H[:,n-1] = AL[:,idCol]
        XD[n-1] = vecXTrue[idCol]

    ALri = np.append(np.append(2*F.real,-2*F.imag,axis=1),H.real,axis=1)
    xLri = np.append(np.append(XT1.real,XT1.imag),XD.real)
    d = np.append(np.zeros(F.shape[1]*2),np.ones(N))

    return ALri,d

def compGradient(xLri,ALri,gamma,d,b):
    print(ALri.shape)
    exit()


    g = 2*(ALri.T)@ALri@xLri-2*(ALri.T)@b+(gamma*d).reshape(-1,1)
    return g

def compuStepSize(xLri,g,ALri,gamma,d,b):
    y = ALri@xLri
    t = float((2*((y-b).T)@ALri@g+gamma*(d.T)@g)/(2*(g.T)@(ALri.T)@ALri@g))
    return t

def convert_xLri2X(xLri,N):
    XD = xLri[(-1-N+1):]
    nRemain = (len(xLri)-N)//2
    real_XT1 = xLri[:nRemain]
    imag_XT1 = xLri[nRemain:2*nRemain]
    XT1 = real_XT1 + 1j*imag_XT1
    vecX = np.zeros((N**2,1)).astype(np.complex128)
    nd = 0


    for n in range(N):
        idDiag = n*(N+1)
        vecX[idDiag] = XD[n]/2
        idLower = np.arange(n+1,N)+(n)*N
        id = nd + np.arange(0,len(idLower))
        vecX[idLower] = XT1[id]
        nd = nd + len(idLower)


    vecX[-1] = XD[N-1]

    X1 = vecX.T.reshape((N,N)).T
    X2 = X1.conj().T
    X = X1 + X2

    return X

def algPhaseLift(ALri,d,b):
    N = int(np.sqrt(ALri.shape[1]))

    # iterative retrieval
    maxIte = int(1e4)
    gamma = 0.5

    i_xLri = np.random.uniform(-20,20,size=(ALri.shape[1],1))
    # save('i_xLri','i_xLri')
    # load i_xLri.mat;

    thre_absErr = 1e-3
    thre_relErr = 1e-5
    thre_relErr_xEst = 1e-5

    vec_absErr = np.zeros((maxIte,1))
    vec_relErr = np.ones((maxIte,1))
    vec_relErr_xEst = np.ones((maxIte,1))
    recSol_xEst = np.zeros((ALri.shape[1],maxIte))
    for iter in range(maxIte):
        if iter%100 == 0:
            loadbar(iter,maxIte)
        g = compGradient(i_xLri,ALri,gamma,d,b)
        t = compuStepSize(i_xLri,g,ALri,gamma,d,b)
        i_xLri = i_xLri - t*g

        recSol_xEst[:,iter] = i_xLri.flatten()
        vec_absErr[iter] = np.linalg.norm(b-ALri@i_xLri)**2

        if iter > 0:
            vec_relErr[iter] = np.abs(vec_absErr[iter]-vec_absErr[iter-1])/vec_absErr[iter]
            vec_relErr_xEst[iter] = np.max(np.abs(recSol_xEst[:,iter]-recSol_xEst[:,iter-1])/np.abs(recSol_xEst[:,iter]))

        if (vec_absErr[iter] < thre_absErr) or (vec_relErr[iter] < thre_relErr) or (vec_relErr_xEst[iter] < thre_relErr_xEst):
            vec_absErr = vec_absErr[:(iter+1)]
            vec_relErr = vec_relErr[:(iter+1)]
            vec_relErr_xEst = vec_relErr_xEst[:(iter+1)]
            recSol_xEst = recSol_xEst[:,:(iter+1)]
            break

    plt.semilogy(np.arange(0,len(vec_absErr)),vec_absErr)
    plt.show()

    plt.semilogy(np.arange(0,len(vec_relErr)),vec_relErr)
    plt.show()

    plt.semilogy(np.arange(0,len(vec_relErr_xEst)),vec_relErr_xEst)
    plt.show()

    X = convert_xLri2X(i_xLri,N)
    D,V = np.linalg.eig(X)
    xEst = np.sqrt(D[0])*V[:,0]

    return xEst

if __name__ == '__main__':
    mat_am, b, xTrue = generateSimulatedField()

    ALri, d = compoMatrix4Inv(mat_am,xTrue)

    xEst = algPhaseLift(ALri,d,b)

    normalized_xTrue = xTrue*np.conj(xTrue[0])/np.abs(np.conj(xTrue[0]))
    normalized_xEst = xEst*np.conj(xEst[0])/np.abs(np.conj(xEst[0]))
    N0 = (len(xEst)-1)/2
    vecn = np.arange(-N0,N0+1)

    plt.plot(vecn,np.angle(normalized_xTrue))
    plt.plot(vecn,np.angle(normalized_xEst),'*')
    plt.xlim([-12, 12])
    plt.show()

    plt.plot(vecn,np.abs(normalized_xTrue))
    plt.plot(vecn,np.abs(normalized_xEst),'*')
    plt.xlim([-12, 12])
    plt.show()
