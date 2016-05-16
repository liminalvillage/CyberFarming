

from saliencymap import Dij, estimateCenter, gD, gauss
from localmax import getLocalMax, convertIndices
import matplotlib.pyplot as plt
import cv2
from os.path import isfile, join
from os import listdir
from computecorners import makePairs, affineTransform
from parameters import setHoG, getParameters, showHoG

import numpy as np
from matplotlib import pyplot as plt
from numpy import arange, sqrt, exp, pi, meshgrid, arctan, zeros, ceil
from scipy.ndimage import convolve1d, convolve
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os.path import isfile, join
from os import listdir
from skimage import io, color

from skimage.feature import hog
from skimage import data, color, exposure

def getImages(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n]= cv2.imread(join(mypath, onlyfiles[n]))
    return images

def getSaliency(img, firstKernel, secondKernel):
    Dx,Dy = Dij(img, firstKernel) # get displacement x and y
    # get image with saliency
    saliency, blurredSaliency = estimateCenter(Dx,Dy, secondKernel)
    return blurredSaliency

def displayThis(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()




def extractFeatures(image, indices, kernel, imageNum, M, N):
    bs = 200
    borderReplicate = cv2.copyMakeBorder(image, bs,bs,bs,bs,cv2.BORDER_REPLICATE)
    newIndices = convertIndices(indices,bs)
    pairs = makePairs(newIndices) # make pairs of all the indices
    parametersList = np.array([])
    hight = img.shape[0]
    width = img.shape[1]
    #print img.shape
    #print len(pairs)
    amount = len(pairs)
    for p in range(0, amount):
        pair = pairs[p]
    #for pair in pairs:
    #    possible = True # default is true
        # check if affine transformation is possible
        #for coordinates in pair:
        #    if coordinates[0] < 0 or coordinates[0] > hight or coordinates[1] < 0 or coordinates[1] > width:
        #        possible = False # if coordinate is negative set to false
        #        # break if one coordinate is negative, no need to check the rest
        #        print pair
        #        break
        #if possible: # if all coordinates of the corners are within the image
        #counter = counter + 1
        yaxis = pair[:, :-1] # y coordinates
        xaxis = pair[:, -1:] # x coordinates
        # affine transformation to cut the feature area out of the original
        # image, all are set to the same size (128,64)
        extractedArea = affineTransform(borderReplicate, xaxis[0],yaxis[0],xaxis[1],
                                        yaxis[1],xaxis[2],yaxis[2], M,N)
        smoothedExtArea = convolve(extractedArea, gauss(kernel), mode='nearest')
        #displayThis(smoothedExtArea)
        plt.imshow(smoothedExtArea, cmap=cm.gray)
        plt.imsave('./output/saliency3/extfeat2/ara2013_extracted%d_plant%d.png' %(p,imageNum), extractedArea, cmap=cm.gray)
        hog = setHoG(smoothedExtArea)
        parameters = getParameters(smoothedExtArea, hog)
        #showHoG(smoothedExtArea,parameters)
        # use p as the id of the feature
        parameters = np.insert(parameters, 0, p, 0)
        if p == 0:
            parametersList = np.array(parameters)
        else:
            parametersList= np.hstack((parameters, parametersList))
    print p
    #print parametersList
    np.savetxt("./output/saliency3/extfeat2/features_plant%d.csv" %imageNum, parametersList, delimiter=",")




if __name__ == '__main__':
    # path to the stored images
    mypath = '../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_rgb_img'
    images = getImages(mypath)

    for i in range(0,10):
        img = images[i]
        copy = img # copy to print the img with the saliencyPoints
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray
        #displayThis(img)
        DijKernel = 3 # kernel to get the displacement coordinates
        saliencyKernel = 7 # kernel to get the smoothed saliency img
        saliency = getSaliency(img, DijKernel, saliencyKernel)
        saliencyPoints = getLocalMax(saliency)
        N = 128 # hight of the extracted feature image
        M = 64 # width of the extracted feature image
        featureSmoothing = 3 # kernel to smooth the extracted area
        extractFeatures(img, saliencyPoints, featureSmoothing, i, M, N) # use origina image for feature extraction
        plt.imshow(saliency,cmap=cm.gray)
        plt.imsave('./output/saliency3/ara2013_dk%d_sk%d_plant%d.png' %(DijKernel, saliencyKernel,i), saliency, cmap=cm.gray)
        for j in saliencyPoints:
            y,x = j.ravel()
            cv2.circle(copy, (x,y), 2, 255, -1)
        plt.imshow(copy, cmap=cm.gray)
        plt.imsave('./output/saliency3/ara2013_dk%d_sk%d_withpoints_plant%d.png' %(DijKernel, saliencyKernel,i), copy, cmap=cm.gray)


    #onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
    #images = np.empty(len(onlyfiles), dtype=object)
    #for n in range(0, len(onlyfiles)):
    #    images[n]= cv2.imread(join(mypath, onlyfiles[n]))
    """
    for i in range(0, 10):
    #for i in range(0, len(images)):
        img = images[i]
        #img = cv2.imread('../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_rgb_img/ara2013_plant001_rgb.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = 3
        Dx,Dy = Dij(gray, kernel)
        kernel = 7
        saliency, blurredSaliency = estimateCenter(Dx,Dy, kernel)
        # save images
        plt.imshow(blurredSaliency,cmap=cm.Greys)
        plt.imsave('./output/saliency/ara2013bs_plant%d.png' %i, blurredSaliency, cmap=cm.Greys)
        #inputImg = np.transpose(np.transpose(blurredSaliency)[0])
        #print blurredSaliency
        #print inputImg
        inputImg = blurredSaliency
        kernel = 3
        Lx = gD(inputImg, kernel, 1, 0)
        Ly = gD(inputImg, kernel, 0, 1)
        G = sqrt((Lx*Lx)+(Ly*Ly))
        pointIndices = getLocalMax(G)
        print pointIndices

        for j in pointIndices:
            y,x = j.ravel()
            cv2.circle(inputImg, (x,y), 3, 255, -1)
        print inputImg
        plt.imshow(blurredSaliency, cmap=cm.Greys)
        plt.show()
    """


"""
mypath = './output/saliency'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n]= cv2.imread(join(mypath, onlyfiles[n]))

for i in range(0, 10):
    img = images[i]
    # transpose to get the first column, transpose back to correct shape
    img = np.transpose(np.transpose(img)[0])
    #print img
    #print img.shape
    #print np.transpose(np.transpose(img)[0])
    #print np.transpose(np.transpose(img)[0]).shape
    #plt.imshow(newimg, cmap=cm.Greys)
    #plt.show()
    kernel = 3
    # gradient in each position
    Lx = gD(img, kernel, 1, 0)
    Ly = gD(img, kernel, 0, 1)
    G = sqrt((Lx*Lx)+(Ly*Ly))
    pointIndices = getLocalMax(G)
"""
