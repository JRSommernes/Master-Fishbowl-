import numpy as np
from time_reversal import *
from constants import *
import matplotlib.pyplot as plt
from PIL import Image
import os
from find_resolution_limit import *

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
        dipoles += '[{0:.6f} {1:.6f} {2:.6f}]'.format(dipole_pos[k][0]/lambda_0,dipole_pos[k][1]/lambda_0,dipole_pos[k][2]/lambda_0)

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
        dipoles += '[{0:.6f} {1:.6f} {2:.6f}]'.format(dipole_pos[i][0]/lambda_0,dipole_pos[i][1]/lambda_0,dipole_pos[i][2]/lambda_0)

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

def reconstruct_image(pos, pol, subdir, subsubdir, FoV, N_sen = 300, N_recon = 100):
    microscope = Microscope(N_sen,N_recon,FoV)

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

def coverge_resolution_limit(subdir,subsubdir,N_sensors,FoV,N_reconstruction):
    carryOn = True
    path = 'C:/Python/Master (Fishbowl)/Images/'+subdir+'/'+subsubdir+'/2_dipoles__{}_sensors'.format(N_sensors)
    if subsubdir == 'Symmetric_around_0':
        offset = 0
    else:
        offset = 1
    if subdir == 'Orthogonal_dipoles':
         polarization = np.array([[1,0,0],[0,0,1]])
         dist_1 = 0.62*lambda_0
         dist_2 = 0.64*lambda_0
    elif subdir == 'Parallel_dipoles':
        polarization = np.array([[1,0,0],[1,0,0]])
        dist_1 = 0.96*lambda_0
        dist_2 = 0.98*lambda_0

    x_1 = dist_1/2
    dipole_pos_1 = np.array([[-x_1,offset*lambda_0,0],[x_1,offset*lambda_0,0]])
    x_2 = dist_2/2
    dipole_pos_2 = np.array([[-x_2,offset*lambda_0,0],[x_2,offset*lambda_0,0]])

    stack_1 = reconstruct_image(dipole_pos_1,polarization,subdir,subsubdir, FoV, N_sen=N_sensors, N_recon=N_reconstruction)
    r_1 = find_modified_rayleigh(stack_1)
    stack_2 = reconstruct_image(dipole_pos_2,polarization,subdir,subsubdir, FoV, N_sen=N_sensors, N_recon=N_reconstruction)
    r_2 = find_modified_rayleigh(stack_2)

    while carryOn:
        dist = (dist_1+dist_2)/2
        x = dist/2
        dipole_pos = np.array([[-x,offset*lambda_0,0],[x,offset*lambda_0,0]])
        stack = reconstruct_image(dipole_pos,polarization,subdir,subsubdir, FoV, N_sen=N_sensors, N_recon=N_reconstruction)
        r = find_modified_rayleigh(stack)

        err = r - 0.735
        if abs(err) < 1e-4:
            carryOn = False

        if r < 0.735:
            r_2 = r
            x_2 = x
            dist_2 = dist
        else:
            r_1 = r
            x_1 = x
            dist_1 = dist


subdirect = 'Orthogonal_dipoles'
subsubdirect = 'Symmetric_around_0'
N_reconstruction = 100
# N_sensors = 100
FoV = 4*lambda_0

for N_sensors in [100,200,400,600,800]:
    coverge_resolution_limit(subdirect,subsubdirect,N_sensors,FoV,N_reconstruction)





# N_reconstruction = 100
# FoV = 4*lambda_0
#
# for N_sensors in [100,200,300,400,500]:
# # for N_sensors in [40,50,60,70,80,90]:
#     subdirect = ['Orthogonal_dipoles','Parallel_dipoles']
#     polarization = np.array([[[1,0,0],[0,0,1]],[[1,0,0],[1,0,0]]])
#     for i, pol in enumerate(polarization):
#         subdir = subdirect[i]
#         subsubdir = ['Symmetric_around_0', 'Symmetric_off_center']
#         for j in range(len(subsubdir)):
#             if subdir == 'Orthogonal_dipoles':
#                 dist = np.linspace(0.625*lambda_0,0.64*lambda_0,20)
#             else:
#                 dist = np.linspace(0.96*lambda_0,0.975*lambda_0,20)
#             for num in dist:
#                 x = num/2
#                 dipole_pos = np.array([[-x,j*lambda_0,0],[x,j*lambda_0,0]])
#                 if check_directory(subdir,subsubdir[j],dipole_pos,N_sensors) == True:
#                     continue
#                 reconstruct_image(dipole_pos,pol,subdir,subsubdir[j], FoV, N_sen=N_sensors, N_recon=N_reconstruction)
#
# FoV = 16*lambda_0
#
# subdirect = ['Single dipole']
# polarization = np.array([[[1,0,0]]])
# sensor_ammount = np.array([300,500,700,900,1000])
# for m in sensor_ammount:
#     for i, pol in enumerate(polarization):
#         subdir = subdirect[i]
#
#         subsubdir = ['FoV edge test']
#
#         for j in range(len(subsubdir)):
#             r = np.linspace(0*lambda_0,5*lambda_0,15)
#             for k in r:
#                 dipole_pos = np.array([[k,k,0]])
#                 if check_directory(subdir,subsubdir[j],dipole_pos,m) == True:
#                     continue
#
#                 reconstruct_image(dipole_pos,pol,subdir,subsubdir[j],FoV,N_sen = m)




# dipole_pos = np.array([[-0.3*lambda_0,0.7*lambda_0,0],[0.7*lambda_0,0.7*lambda_0,0]])
# polarization = np.array([[1,0,0],[0,0,1]])

# plt.imshow(microscope.I[:,:,N_reconstruction//2],extent=(-FoV/2,FoV/2,FoV/2,-FoV/2))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.colorbar()
# for dipole in microscope.dipoles:
#     plt.scatter(dipole.x,dipole.y,color='r')
# plt.show()
