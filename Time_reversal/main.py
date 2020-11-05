import numpy as np
from time_reversal import *
from constants import *
import matplotlib.pyplot as plt
from PIL import Image
import os
from find_resolution_limit import *

from mpl_toolkits.mplot3d import Axes3D

def check_directory(subdir,subsubdir,dipole_pos,N_sensors):
    folder = 'C:/Python/Master (Fishbowl)/images'
    directory = folder+'/'+subdir+'/'+subsubdir+'/{}_dipoles__{}_sensors'.format(len(dipole_pos),N_sensors)
    try:
        names = os.listdir(directory)
    except:
        return False
    dipoles = ''
    for k in range(len(dipole_pos)):
        if k != 0:
            dipoles += '__'
        dipoles += '[{0:.8f} {1:.8f} {2:.8f}]'.format(dipole_pos[k][0]/lambda_0,dipole_pos[k][1]/lambda_0,dipole_pos[k][2]/lambda_0)

    if dipoles in names:
        print(dipoles+' already in directory')
        return True
    else:
        return False

def saveimage(I,dipole_pos,subdir,subsubdir,FoV,N_sensors=300,N_recon=100):
    folder = 'C:/Python/Master (Fishbowl)/images'
    names = os.listdir(folder)

    if subdir not in names:
        os.mkdir(folder+'/'+subdir)

    os.chdir(folder+'/'+subdir)

    names = os.listdir()

    if subsubdir not in names:
        os.mkdir(subsubdir)

    os.chdir(subsubdir)

    names = os.listdir()

    setup = '{}_dipoles__{}_sensors'.format(len(dipole_pos),N_sensors)

    if setup not in names:
        os.mkdir(setup)

    os.chdir(setup)
    dipoles = ''
    for i in range(len(dipole_pos)):
        if i != 0:
            dipoles += '__'
        dipoles += '[{0:.8f} {1:.8f} {2:.8f}]'.format(dipole_pos[i][0]/lambda_0,dipole_pos[i][1]/lambda_0,dipole_pos[i][2]/lambda_0)

    names = os.listdir()
    if dipoles in names:
        print(dipoles+' already in directory')
        # os.chdir('C:/python/Master (Fishbowl)/class_time_reversal')
        return
    else:
        os.mkdir(dipoles)

    for i in range(N_recon):
        im = Image.fromarray(I[:,:,i].astype(np.float64))
        im.save(dipoles+'/{}.tiff'.format(i))

    plt.imshow(I[:,:,N_recon//2],extent=(-FoV/2,FoV/2,FoV/2,-FoV/2))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    for dipole in dipole_pos:
        plt.scatter(dipole[0],dipole[1],color='r')
    plt.savefig(dipoles+'.tiff')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure

    os.chdir('C:/Python/Master (Fishbowl)')

def reconstruct_image(pos, pol, subdir, subsubdir, FoV, k_0, N_sen = 300, N_recon = 100):
    microscope = Microscope(N_sen,N_recon,FoV,k_0)

    microscope.make_sensors(sensor_radius)
    microscope.make_dipoles(pos,pol)
    microscope.reconstruct_image(N_recon)

    saveimage(microscope.I,pos,subdir,subsubdir,FoV,N_sen,N_recon)

    return microscope.I

def find_modified_rayleigh(stack):
    x,y,z = np.shape(stack)
    line = stack[np.where(stack==np.amax(stack))[0][0],:,z//2]
    maxima, maxima_idx, minima, minima_idx = find_extremum(line)

    background = np.amin(line)
    diff = ((minima-background)/(np.amin(maxima)-background))

    return diff

def coverge_resolution_limit(subdir,subsubdir,N_sensors,FoV,N_reconstruction, k_0):
    carryOn = True
    path = 'C:/Python/Master (Fishbowl)/Images/'+subdir+'/'+subsubdir+'/2_dipoles__{}_sensors'.format(N_sensors)
    if subsubdir == 'Symmetric_around_0':
        offset = 0
    else:
        offset = 1
    if subdir == 'Orthogonal_dipoles' or subsubdir == 'Orthogonal_dipoles' or subdir == 'Different_wavelengths_orthogonal_dipoles':
         polarization = np.array([[1,0,0],[0,0,1]])
         dist_1 = 0.5*lambda_0
         dist_2 = 1*lambda_0
    elif subdir == 'Parallel_dipoles' or subsubdir == 'Parallel_dipoles':
        polarization = np.array([[1,0,0],[1,0,0]])
        dist_1 = 0.9*lambda_0
        dist_2 = 2.5*lambda_0

    x_1 = dist_1/2
    dipole_pos_1 = np.array([[-x_1,offset*lambda_0,0],[x_1,offset*lambda_0,0]])
    x_2 = dist_2/2
    dipole_pos_2 = np.array([[-x_2,offset*lambda_0,0],[x_2,offset*lambda_0,0]])

    stack_1 = reconstruct_image(dipole_pos_1,polarization,subdir,subsubdir, FoV, k_0, N_sen=N_sensors, N_recon=N_reconstruction)
    r_1 = find_modified_rayleigh(stack_1)
    stack_2 = reconstruct_image(dipole_pos_2,polarization,subdir,subsubdir, FoV, k_0, N_sen=N_sensors, N_recon=N_reconstruction)
    r_2 = find_modified_rayleigh(stack_2)


    while carryOn:
        dist = (dist_1+dist_2)/2
        x = dist/2
        dipole_pos = np.array([[-x,offset*lambda_0,0],[x,offset*lambda_0,0]])
        stack = reconstruct_image(dipole_pos,polarization,subdir,subsubdir, FoV, k_0, N_sen=N_sensors, N_recon=N_reconstruction)
        r = find_modified_rayleigh(stack)

        err = r - 0.735
        print(abs(err),r)
        if abs(err) < 1e-5:
            carryOn = False
        if abs(r) < 0.735:
            r_2 = r
            x_2 = x
            dist_2 = dist
        else:
            r_1 = r
            x_1 = x
            dist_1 = dist



# subdirect = 'Orthogonal_dipoles'
# subsubdirect = 'Symmetric_around_0'
# N_reconstruction = 100
# FoV = 4*lambda_0
# N_sensors = 300

# subdirect = 'Different_wavelengths_orthogonal_dipoles'
#
# N_reconstruction = 100
# N_sensors = 300


#DONE
#100,110,120,130,140,150,160,170,180,190,200,250,300,350,400,450,500,550,600,650,700,750,800,101,151,201,251,301,351,401,451,
# for lambda_0 in np.linspace(400,750,10)*1e-9:
#     subsubdirect = '{}_nm'.format(lambda_0*1e9)
#     sensor_radius = 10*lambda_0
#
#     freq = c_0/lambda_0
#     k_0 = 2*np.pi*freq*np.sqrt(eps_0*mu_0)
#     omega = 2*np.pi*freq
#     FoV = 4*lambda_0
#     coverge_resolution_limit(subdirect,subsubdirect,N_sensors,FoV,N_reconstruction,k_0)
#

# subdirect = 'Orthogonal_dipoles'
# subsubdirect = 'Symmetric_around_0'
# N_reconstruction = 100
# # N_sensors = 100
# FoV = 4*lambda_0
# for N_sensors in [20,21,24,25,30,31,34,35,40,41,50,51,60,61,70,71,80,81,90,91,105,115,125,135,145,155,165,175,185,195,220,225,270,275,320,325,370,375,420,425,470,475,520,525,570,575,620,625,670,675,720,725,770,775]:
#     coverge_resolution_limit(subdirect,subsubdirect,N_sensors,FoV,N_reconstruction,k_0)

subdirect = 'Parallel_dipoles'
subsubdirect = 'Symmetric_around_0'
N_reconstruction = 100
# N_sensors = 100
FoV = 4*lambda_0
for N_sensors in [20,21,24,25,30,31,34,35,40,41,50,51,60,61,70,71,80,81,90,91,105,115,125,135,145,155,165,175,185,195,220,225,270,275,320,325,370,375,420,425,470,475,520,525,570,575,620,625,670,675,720,725,770,775]:
    coverge_resolution_limit(subdirect,subsubdirect,N_sensors,FoV,N_reconstruction,k_0)

# paths = find_direcories()
# res = []
# dist = []
# plot_resolution_limit(paths)


# N_reconstruction = 100
# FoV = 4*lambda_0
# N_sensors = 500
#
# pol = np.array([[1,0,0],[0,0,1]])
# subdir = 'Orthogonal_dipoles'
# subsubdir = 'Symmetric_around_0'
#
# dist = np.linspace(0.2*lambda_0,1.5*lambda_0,10)
# for num in dist:
#     x = num/2
#     dipole_pos = np.array([[-x,0,0],[x,0,0]])
#     if check_directory(subdir,subsubdir,dipole_pos,N_sensors) == True:
#         continue
#     reconstruct_image(dipole_pos,pol,subdir,subsubdir, FoV, N_sen=N_sensors, N_recon=N_reconstruction)




# path = 'C:/Python/Master (Fishbowl)/Images/Orthogonal_dipoles/Symmetric_around_0/2_dipoles__500_sensors'
# find_reyleigh(path)
# plot_resolution_limit()
