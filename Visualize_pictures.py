
from PIL import Image
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import cv2




def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getimage(folder_path):
    # get dirpath , subdirpath , filename
    filelist = list(os.walk(folder_path))[0]

    # get batch file
    batchlist = [filelist[0] + '/'+i for i in filelist[-1] if '_' in i]

    # get data
    data = [ unpickle(i) for i  in batchlist  ]

    return  data

images = getimage('./data/cifar-10-batches-py')
''''
http://www.cs.toronto.edu/~kriz/cifar.html
batches.meta. It too contains a Python dictionary object. 
It has the following entries:
label_names -- a 10-element list which gives 
meaningful names to the numeric labels in the labels 
array described above. For example, 
label_names[0] == "airplane", label_names[1] == "automobile", etc.

print(images[0].keys())

dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
** for every batch file
    data -- a 10000x3072 numpy array of uint8s. 
    Each row of the array stores a 32x32 colour image. 
    The first 1024 entries contain the red channel values,
    the next 1024 the green, and the final 1024 the blue. 
    The image is stored in row-major order, 
    so that the first 32 entries of the array are the red channel 
    values of the first row of the image.

**
    labels -- a list of 10000 numbers in the range 0-9. 
    The number at index i indicates the label of the ith 
    image in the array data.

'''
def printimages(imagedatalist : list) -> None:
    batch1items = imagedatalist[0]
    batch1data = batch1items[b'data']
    # print(batch1data.shape) # (10000, 3072)
    batch1labels = np.array(batch1items[b'labels'])
    index =  random.sample(range(len(batch1data)), 9)
    image = batch1data[index,:]
    labels = batch1labels[index]
    # Create four polar axes and access them through the returned array
    fig, axs = plt.subplots(3, 3)
    k = 0
    for i in range(3):
        for j in range(3):
            R = np.array(image[k,:1024]).reshape(32,32)
            G = np.array(image[k,1024:1024*2]).reshape(32,32)
            B = np.array(image[k,1024*2:]).reshape(32,32)
            axs[i,j].imshow(cv2.merge((B,G,R)))
            axs[i,j].set_xlabel(labels[k])
            k += 1
    fig.tight_layout() # 调整间距
    plt.show()

printimages(images)












