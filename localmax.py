##########################################################
# Iris Verweij
# May 4, 2016
#
# find local maxima
##########################################################

import cv2
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

def derivF(X, s, order):
    """
    derivF
    X       : array of points on which to take the derivative
    s       : scale of the gaussian
    order   : order of the derivative

    returns
    array  with the gaussian derivatives of scale s at the points in X

    Gaussian derivative up to 2nd order
    """
    if order == 0:
        G = (1/(s*sqrt(2*pi))*exp(-((X**2)/(2*(s**2)))))
    elif order == 1:
        G = (1/(s*sqrt(2*pi))*exp(-((X**2)/(2*(s**2))))*-(X/(s**2)))
    elif order == 2:
        G = (1/(s*sqrt(2*pi))*exp(-((X**2)/(2*(s**2))))*(((X**2)
             - (s**2))/(s**4)))
    return G


def gD(F, s, iorder, jorder):
    """
    gD
    F       : image to convolve
    s       : scale of the gaussian
    iorder  : i order of the derivative
    jorder  : j order of the derivative

    returns
    image convolved with the gaussian derivative

    Convolve image with gaussian derivatives
    """
    F = F.astype(float)
    s = float(s)
    S = ceil(s*3)
    x = arange(float(0-S), float(1+S)).astype(float)
    y = arange(float(0-S), float(1+S)).astype(float)
    convolved = convolve1d(convolve1d(F, derivF(x, s, iorder), axis=1,
                           mode='nearest'), derivF(y, s, jorder), axis=0,
                           mode='nearest')
    return convolved


def getThreshold(maxima):
    std = np.std(maxima)
    mean = np.mean(maxima)


    # threshold is based on the highest mean standard deviation
    stdx = np.mean(np.std(G, axis=1))
    stdy = np.mean(np.std(G, axis=0))
    meanstd = (stdx+stdy)/2
    print meanstd
    #if stdx <= stdy:
    #    std = stdy
    #else:
    #    std = stdx

    miniy = min(np.amin(G, axis=0))
    minix = min(np.amin(G, axis=1))
    maxiy = max(np.amax(G, axis=0))
    maxix = max(np.amax(G, axis=1))
    if miniy < minix:
        mini = miniy
    else:
        mini = minix
    if maxiy > maxix:
        maxi = maxiy
    else:
        maxi = maxix
    threshold = (((maxi-meanstd) + (mini-meanstd))/2)
    print stdy, stdx, threshold
    return threshold

def isMax(G, y, x):
    zero = G[y][x]
    one = G[y+1][x]
    two = G[y-1][x]
    three = G[y][x+1]
    four = G[y][x-1]
    five = G[y+1][x+1]
    six = G[y+1][x-1]
    seven = G[y-1][x+1]
    eight = G[y-1][x-1]
    if (zero > one and zero > two and zero > three and zero > four
    and zero > five and zero > six and zero > seven and zero > eight):
        value = True
    elif (zero < one and zero < two and zero < three and zero < four
    and zero < five and zero < six and zero < seven and zero < eight):
        value = True
    else:
        value = False
    return value

def getLocalMax(G):
    width = G.shape[1]
    hight = G.shape[0]
    #threshold = getThreshold(G)
    threshold = 0.1
    values = []
    maxima = np.zeros((hight,width))
    for x in range(0,width-1):
        for y in range(0,hight-1):
            # if local maxima
            if (isMax(G,y,x)):
                maxima[y][x] = G[y][x]
                values.append(G[y][x])
            #zero = G[y][x]
            #one = G[y+1][x]
            #two = G[y-1][x]
            #three = G[y][x+1]
            #four = G[y][x-1]
            #five = G[y+1][x+1]
            #six = G[y+1][x-1]
            #seven = G[y-1][x+1]
            #eight = G[y-1][x-1]
            #if (zero > one and zero > two and zero > three and zero > four
            #and zero > five and zero > six and zero > seven and zero > eight):
            #    maxima[y][x] = G[y][x]
            #    values.append(G[y][x])

    #print maxima
    #print values
    #print np.std(values)
    mean = np.mean(values)
    std = np.std(values)
    meanstd = std/len(values)
    print mean
    if mean < std:
        threshold = mean - meanstd
    else:
        threshold = std - meanstd
    #threshold = np.mean(values)
    nonzero = np.transpose(np.nonzero(maxima))
    for i in nonzero:
        if maxima[i[0]][i[1]] < threshold:
            maxima[i[0]][i[1]] = 0
    indices = np.transpose(np.nonzero(maxima))

    # indices of the points to extract features
    #nonzero = np.transpose(np.nonzero(maxima))
    #plt.imshow(maxima, cmap=cm.Greys)
    #plt.show()
    return indices



if __name__=="__main__":
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

        for j in pointIndices:
            y,x = j.ravel()
            cv2.circle(images[i], (x,y), 3, 255, -1)
        plt.imshow(img, cmap=cm.Greys)
        plt.show()


    """
        #img = cv2.imread('../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_rgb_img/ara2013_plant001_rgb.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = 3
        Dx,Dy = Dij(gray, kernel)
        kernel = 7
        saliency, blurredSaliency = estimateCenter(Dx,Dy, kernel)
        # save images
        plt.imshow(blurredSaliency,cmap=cm.Greys)
        plt.imsave('./output/saliency/ara2013bs_plant%d.png' %i, blurredSaliency, cmap=cm.Greys)


    # print the images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(gray, cmap=cm.Greys)
    ax1.set_title('Input img')
    ax1.set_adjustable('box-forced')
    ax2.axis('off')
    ax2.imshow(saliency, cmap=cm.Greys)
    ax2.set_title('Saliency before Blurr')
    ax2.set_adjustable('box-forced')
    ax3.axis('off')
    ax3.imshow(blurredSaliency, cmap=cm.Greys)
    ax3.set_title('Blurred Saliency')
    ax3.set_adjustable('box-forced')
    plt.show()
    """
