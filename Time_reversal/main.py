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


def coverge_resolution_limit(subdir,subsubdir,N_sensors,FoV,N_reconstruction):
    #DO THIS BETTER
    carryOn = True
    path = 'C:/Python/Master (Fishbowl)/Images/'+subdir+'/'+subsubdir+'/2_dipoles__{}_sensors'.format(N_sensors)
    if subsubdir == 'Symmetric_around_0':
        offset = 0
    else:
        offset = 1
    if subdir == 'Orthogonal_dipoles':
         polarization = np.array([[[1,0,0],[0,0,1]]])
         dist_1 = 0.62*lambda_0
         dist_2 = 0.64*lambda_0
    elif subdir == 'Parallel_dipoles':
        polarization = np.array([[[1,0,0],[1,0,0]]])
        dist_1 = 0.96*lambda_0
        dist_2 = 0.98*lambda_0

    x_1 = dist_1/2
    dipole_pos_1 = np.array([[-x_1,offset*lambda_0,0],[x_1,offset*lambda_0,0]])
    x_2 = dist_2/2
    dipole_pos_2 = np.array([[-x_2,offset*lambda_0,0],[x_2,offset*lambda_0,0]])

    if check_directory(subdir,subsubdir[j],dipole_pos_1,N_sensors) == True:
        continue
    if check_directory(subdir,subsubdir[j],dipole_pos_2,N_sensors) == True:
        continue

    reconstruct_image(dipole_pos_1,pol,subdir,subsubdir[j], FoV, N_sen=N_sensors, N_recon=N_reconstruction)
    reconstruct_image(dipole_pos_2,pol,subdir,subsubdir[j], FoV, N_sen=N_sensors, N_recon=N_reconstruction)

    dipoles_1 = ''
    for i in range(len(dipole_pos_1)):
        if i != 0:
            dipoles_1 += '__'
        dipoles_1 += '[{0:.6f} {1:.6f} {2:.6f}]'.format(dipole_pos_1[i][0]/lambda_0,dipole_pos_1[i][1]/lambda_0,dipole_pos_1[i][2]/lambda_0)

    dipoles_2 = ''
    for i in range(len(dipole_pos_2)):
        if i != 0:
            dipoles_2 += '__'
        dipoles_2 += '[{0:.6f} {1:.6f} {2:.6f}]'.format(dipole_pos_2[i][0]/lambda_0,dipole_pos_2[i][1]/lambda_0,dipole_pos_2[i][2]/lambda_0)

    while carryOn:
        stack_1 = np.zeros((N_reconstruction,N_reconstruction,N_reconstruction))
        stack_2 = np.zeros((N_reconstruction,N_reconstruction,N_reconstruction))
        for i in range(z):
            stack_1[:,:,i] = Image.open(path+'/'+dipoles_1+'/'+'{}.tiff'.format(i))
            stack_2[:,:,i] = Image.open(path+'/'+dipoles_2+'/'+'{}.tiff'.format(i))

        line_1 = stack_1[np.where(stack_1==np.amax(stack_1))[0][0],:,N_reconstruction//2]
        line_2 = stack_2[np.where(stack_2==np.amax(stack_2))[0][0],:,N_reconstruction//2]
        maxima_1, maxima_idx_1, minima_1, minima_idx_1 = find_extremum(line_1)
        maxima_2, maxima_idx_2, minima_2, minima_idx_2 = find_extremum(line_2)

        if 0.8 <= maxima[0]/maxima[1] <= 1.2:
            background = np.amin(line)
            diff = ((minima-background)/(np.amin(maxima)-background))
            if diff <= 0.735:
                if plot_extrema == True:
                    plt.plot(line)
                    plt.plot(maxima_idx,maxima,'*',c='g')
                    plt.plot(minima_idx,minima,'*',c='r')
                    plt.show()
                f.write(path+'/'+element+' ')
                f.write(str(diff)+'\n')
                print(path+'/'+element+' '+str(diff))
                break



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
