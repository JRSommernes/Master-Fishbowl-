import numpy as np
import json

class Fishbowl:
    def __init__(self,N_sensors,radius,wl,n,mur,epsr,k_0,dipoles,M_timepoints):
        self.N_sensors = N_sensors
        self.radius = radius
        self.wl = wl
        self.n_obj = n
        self.mur_obj = mur
        self.epsr_obj = epsr
        self.k_0 = k_0
        self.dipoles = dipoles
        self.M_timepoints = M_timepoints

        self.k_obj = self.k_0*self.n_obj

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

    def limited_aperture_sensors(self,NA):
        self.NA = NA
        self.theta_max = np.arcsin(self.NA/self.n_obj)

        sphere_area = 4*np.pi*self.radius**2
        cap_area = 2*np.pi*self.radius**2*(1-np.cos(self.theta_max))
        mult = sphere_area/cap_area
        N_sensors = self.N_sensors
        self.N_sensors = int(self.N_sensors*mult)

        while 1:
            self.make_sensors()
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

    #Slower if njit because of dyadic green dependecy
    def sensor_field(self,polarizations):
        E_tot = np.zeros((3*self.N_sensors,polarizations.shape[2]),dtype=np.complex128)

        for i in range(len(self.dipoles)):
            G = self.dyadic_green(self.dipoles[i]).transpose(1,0,2)
            G = G.reshape(3*self.N_sensors,3)
            E_tot += G@polarizations[i]

        self.E_stack = E_tot

    #Slower if njit because dependencies
    def data_acquisition(self):
        N_dipoles = len(self.dipoles)

        phi = np.random.uniform(0,2*np.pi,(N_dipoles,self.M_timepoints))
        theta = np.random.uniform(-np.pi/2,np.pi/2,(N_dipoles,self.M_timepoints))

        polarizations = np.array([np.cos(phi)*np.sin(theta),
                                  np.sin(phi)*np.sin(theta),
                                  np.cos(theta)]).swapaxes(0,1)

        self.sensor_field(polarizations)

    # def noise_space(self):
    #     self.E_N = []
    #     for i in range(3):
    #         E_tmp = self.E_stack[i*self.N_sensors:(i+1)*self.N_sensors]
    #         S = E_tmp@np.conjugate(E_tmp).T
    #
    #         eigvals,eigvecs = np.linalg.eig(S)
    #
    #         dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)
    #
    #         noice_idx = np.where(dist<1)[0]
    #         N = len(noice_idx)
    #         D = len(self.E_stack)-N
    #
    #         self.E_N.append(eigvecs[:,noice_idx])

    def find_noise_space(self):
        S = self.E_stack@np.conjugate(self.E_stack).T

        eigvals,eigvecs = np.linalg.eig(S)

        dist = np.sqrt(eigvals.real**2 + eigvals.imag**2)

        noice_idx = np.where(dist<1)[0]
        N = len(noice_idx)
        D = len(self.E_stack)-N

        E_N = eigvecs[:,noice_idx]

        self.E_N = np.ascontiguousarray(E_N)

    # def check_if_resolvable(self):
    #     self.noise_space()
    #
    #     x1 = self.dipoles[0,0]
    #     x2 = self.dipoles[1,0]
    #
    #     x = np.linspace(x1,x2,20).reshape(-1,1)
    #     y = self.dipoles[0,1]*np.ones_like(x)
    #     z = self.dipoles[0,2]*np.ones_like(x)
    #     pos = np.append(x,np.append(y,z,axis=1),axis=1)
    #
    #
    #     A = np.zeros((len(pos),self.N_sensors,3,3),dtype=np.complex128)
    #     for i in range(len(pos)):
    #         A[i] = self.dyadic_green(pos[i])
    #
    #     P = np.zeros(len(x),dtype=np.complex128)
    #
    #     for i in range(3):
    #         for j in range(3):
    #             for k in range(3):
    #                 P_1 = A[:,:,i,j].conj()@self.E_N[k]
    #                 P_2 = A[:,:,i,j]@self.E_N[k].conj()
    #                 P_t = (1/np.einsum('ij,ij->i',P_1,P_2))
    #                 P+= P_t.reshape(len(x))
    #
    #     for el in P[P.argsort()[-2:]]:
    #         if el not in (P[0],P[-1]):
    #             return False
    #     peak = np.min((P[0],P[-1]))
    #     min = np.min(P)
    #     if min/peak <= 0.735:
    #         return True
    #     else:
    #         return False

    def check_resolvability(self):
        self.find_noise_space()

        x1 = self.dipoles[0,0]
        x2 = self.dipoles[1,0]

        x = np.linspace(x1,x2,20).reshape(-1,1)
        y = self.dipoles[0,1]*np.ones_like(x)
        z = self.dipoles[0,2]*np.ones_like(x)
        pos = np.append(x,np.append(y,z,axis=1),axis=1)


        A = np.zeros((len(pos),self.N_sensors,3,3),dtype=np.complex128)
        for i in range(len(pos)):
            A[i] = self.dyadic_green(pos[i])

        A = A.transpose(2,1,3,0)
        A = A.reshape(3*self.N_sensors,3,len(pos)).T

        a,b,c = A.shape

        A = A.reshape(-1, A.shape[-1])
        B =  self.E_N

        P_1 = np.conjugate(A)@B
        P_2 = A@np.conjugate(B)

        P_1 = P_1.reshape(a,b,P_1.shape[-1])
        P_2 = P_2.reshape(a,b,P_2.shape[-1])

        P = (1/np.einsum('ijk,ijk->ij',P_1,P_2)).T
        P = np.sum(P,axis=0)

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
        self.old_dipoles = np.copy(self.dipoles)*2

        counter=0
        while 1:
            if self.check_resolvability():
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

            self.data_acquisition()
            counter+=1
            print(counter)

        self.dipoles = np.copy(self.old_dipoles)
        self.E_stack = np.copy(self.E_stack_old)

    def dyadic_green_2D(self,xx,yy,zz,grid_size):
        I = np.identity(3)
        shape_1 = np.append(self.N_sensors,np.ones(len(xx.shape),dtype=int))
        shape_2 = np.append(1,xx.shape)

        r_x = self.sensors[0].reshape(shape_1)-xx.reshape(shape_2)
        r_y = self.sensors[1].reshape(shape_1)-yy.reshape(shape_2)
        r_z = self.sensors[2].reshape(shape_1)-zz*np.ones(shape_2)
        r_p = np.array((r_x,r_y,r_z))

        R = np.sqrt(np.sum((r_p)**2,axis=0))
        R_hat = ((r_p)/R)
        RR_hat = np.einsum('iklm,jklm->ijklm',R_hat,R_hat)

        g_R = np.exp(1j*self.k_0*R)/(4*np.pi*R)
        expr_1 = (3/(self.k_0**2*R**2)-3j/(self.k_0*R)-1)*g_R
        expr_1 = np.broadcast_to(expr_1,RR_hat.shape)

        expr_2 = (1+1j/(self.k_0*R)-1/(self.k_0**2*R**2))*g_R
        expr_2 = np.broadcast_to(expr_2,RR_hat.shape)

        I = np.broadcast_to(I,(self.N_sensors,grid_size,grid_size,3,3))
        I = I.transpose(3,4,0,1,2)

        G = (expr_1*RR_hat + expr_2*I).transpose((2,0,1,3,4))

        return G

    def P_calc_2D(self,A_fov,N_recon):
        self.P = np.zeros((N_recon,N_recon),dtype = np.complex128)
        a,b,c,d = A_fov.shape

        A = A_fov.reshape(-1, A_fov.shape[-1])
        B =  self.E_N.reshape(-1, self.E_N.shape[-1])

        P_fov_1 = np.conjugate(A)@B
        P_fov_2 = A@np.conjugate(B)

        P_fov_1 = P_fov_1.reshape(a,b,c,P_fov_1.shape[-1])
        P_fov_2 = P_fov_2.reshape(a,b,c,P_fov_2.shape[-1])

        P_fov_plane = (1/np.einsum('ijkl,ijkl->ijk',P_fov_1,P_fov_2)).T
        self.P += np.sum(P_fov_plane,axis=0)

    def P_estimation(self,N_recon,FoV):
        self.find_noise_space()
        x = y = np.linspace(-FoV/2,FoV/2,N_recon)*self.wl
        xx,yy = np.meshgrid(x,y)
        z = [0]

        A_fov_plane = self.dyadic_green_2D(xx,yy,z,N_recon).transpose(1,0,2,3,4)
        # A_fov_plane = np.ascontiguousarray((A_fov_plane.reshape(3,self.N_sensors,3,N_recon,N_recon)).T)
        A_fov_plane = np.ascontiguousarray(A_fov_plane.reshape(3*self.N_sensors,3,N_recon,N_recon).T)

        self.P_calc_2D(A_fov_plane,N_recon)

    def save_info(self,dir,counter):
        # t0 = round(time())

        try:
            data = {'Resolution limit [wl]' : self.resolution_limit,
                    'N_sensors' : str(self.N_sensors),
                    'Sensor radius' : self.radius,
                    'N_timepoints' : str(self.M_timepoints),
                    'Wavelength' : self.wl,
                    'n_objective' : self.n_obj,
                    'Relative permeability' : self.mur_obj,
                    'epsr_objective' : self.epsr_obj,
                    'NA' : self.NA,
                    'Theta max' : self.theta_max,
                    'Dipole_positions [wl]' : (self.dipoles/self.wl).tolist()}
        except:
            data = {'Resolution limit [wl]' : self.resolution_limit,
                    'N_sensors' : str(self.N_sensors),
                    'Sensor radius' : self.radius,
                    'N_timepoints' : str(self.M_timepoints),
                    'Wavelength' : self.wl,
                    'n_objective' : self.n_obj,
                    'Relative permeability' : self.mur_obj,
                    'epsr_objective' : self.epsr_obj,
                    'Theta max' : 0,
                    'Dipole_positions [wl]' : (self.dipoles/self.wl).tolist()}

        # os.mkdir(dir+'/{}'.format(t0))

        with open(dir+"/{}_data_fishbowl.json".format(counter), 'w') as output:
            json.dump(data, output, indent=4)
