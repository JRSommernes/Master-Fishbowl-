import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

directory = os.listdir('images')

paths = []
for dir in directory:
    if dir != "Single dipole":
        subdirectory = os.listdir('images/'+dir)
        for subdir in subdirectory:
            subsubdirectory = os.listdir('images/'+dir+'/'+subdir)
            for subsubdir in subsubdirectory:
                element = os.listdir('images/'+dir+'/'+subdir+'/'+subsubdir)
                for ele in element:
                    if not ".tiff" in ele:
                        paths.append('images/'+dir+'/'+subdir+'/'+subsubdir+'/'+ele)

for path in paths:
    file = os.listdir(path)
    z = len(file)
    x,y = np.array(Image.open(path+'/'+file[0])).shape

    stack = np.zeros((x,y,z))
    for i in range(z):
        stack[:,:,i] = Image.open(path+'/'+'{}.tiff'.format(i))

    #USED FOR PLOTTING IMAGE STACK
    # for i in range(z):
    #     plt.imshow(stack[:,:,i],vmin=np.amin(stack),vmax=np.amax(stack))
    #     plt.colorbar()
    #     plt.show(block=False)
    #     plt.pause(0.1)
    #     plt.clf()
    #     plt.cla()

    print(path)
    plt.plot(stack[x//2,:,z//2])
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()
    plt.cla()
