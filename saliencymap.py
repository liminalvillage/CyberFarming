#############################################################
# Iris Verweij
# May 2, 2016
#
# Saliency map attempt
#############################################################

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
import numpy.ma as ma



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


def gauss(s):
    """
    gauss
    s       : scale of the gaussian distribution

    returns
    ndarray containing normalized gaussian convolution filter

    2d gaussian convolution filter
    """
    S = ceil(s*3)
    x = arange(float(0-S), float(1+S))
    y = arange(float(0-S), float(1+S))
    Y, X = meshgrid(y, x)
    V = (1/((s**2)*(2*pi)))*exp(-((X**2 + Y**2)/(2*(s**2))))
    return V/sum(sum(V))


def Dij(gray, kernel):
    """
    Dij
    gray    : image in gray scale
    kernel  : size of the gaussian kernel for the convolution

    returns
    2 ndarrays containing the information required for the estimation
    of the most important areas

    Apply Valenti et al. formula for the displacement coordinates
    to estimate the center areas the pixels belong to
    """
    Lx = gD(gray, kernel, 1, 0)
    Ly = gD(gray, kernel, 0, 1)
    Lxx = gD(gray, kernel, 2, 0)
    Lyy = gD(gray, kernel, 0, 2)
    Lxy = gD(gray, kernel, 1, 1)
    D = -(((Lx*Lx)+(Ly*Ly))/(((Ly*Ly)*Lxx)-(2*(Lx*Lxy*Ly))+((Lx*Lx)*Lyy)))
    Dm = D < 0
    D[Dm] = 0
    Dx = np.array((D * Lx)).astype('int')
    Dy = np.array((D * Ly)).astype('int')
    return Dx, Dy

def estimateCenter(Dx, Dy, kernel):
    """
    estimateCenter
    Dx      : displacement coordinates for x location
    Dy      : displacement coordinates for y location
    kernel  : size of the kernel for gaussian blurr

    returns
    ndarray with the saliency before and after the gaussian blurr

    create accumulated image and blurr the points to create points of
    the clusters
    """
    width = Dx.shape[1]
    hight = Dx.shape[0]
    # empty array to fill in with the coordinates
    saliency = np.zeros((hight,width))
    # for each row in the arrays
    for s in range(0,hight):
        rowX = Dx[s]
        rowY = Dy[s]
        # for every column of the row
        for i in range(0,width):
            x = rowX[i]
            y = rowY[i]
            # discard the values that exceed the size of the image
            # threshold based on the norm of the vector
            lengte = np.linalg.norm(np.array([y,x]))
            if lengte > 5:
                if (x+i) < width and (x+i) > 0:
                    if (y+s) < hight and (y+s)>0:
                        saliency[(y+s)][(x+i)] = saliency[(y+s)][(x+i)] + 1
    # Gaussian blurr
    blurredSaliency = convolve(saliency, gauss(kernel), mode='nearest')
    return saliency, blurredSaliency




if __name__=="__main__":
    mypath = '../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_rgb_img'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n]= cv2.imread(join(mypath, onlyfiles[n]))

    for i in range(0, len(images)):
        img = images[i]
        #img = cv2.imread('../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_rgb_img/ara2013_plant001_rgb.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = 3
        Dx,Dy = Dij(gray, kernel)
        kernel = 7
        saliency, blurredSaliency = estimateCenter(Dx,Dy, kernel)
        # save images
        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(8,4) , sharex=True, sharey=True)
        ax2.axis('off')
        ax2.imshow(saliency, cmap=cm.gray)
        ax2.set_title('Saliency before Blurr')
        ax2.set_adjustable('box-forced')
        ax3.axis('off')
        ax3.imshow(blurredSaliency, cmap=cm.gray)
        ax3.set_title('Blurred Saliency')
        ax3.set_adjustable('box-forced')
        plt.show()
        #plt.imsave('./output/saliency/ara2013bs_plant%d.png' %i, blurredSaliency, cmap=cm.Greys)


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
