##############################################################
# Iris Verweij
# First Feature extraction test
# April 25, 2016
###############################################################

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import arange, sqrt, exp, pi, meshgrid, arctan, zeros, ceil
from scipy.ndimage import convolve1d
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os.path import isfile, join
from os import listdir



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

def findTrackingPoints(img,maxCorners, quality, dist):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners, quality, dist)
    corners = np.int0(corners)
    return corners

if __name__=="__main__":
    mypath = '../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_rgb_img'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n]= cv2.imread(join(mypath, onlyfiles[n]))

    for i in range(0, len(images)):
        num = i
        img = images[i]

        corners = findTrackingPoints(img, 25, 0.01, 10)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(img, (x,y), 3, 255, -1)
        plt.imshow(img)
        plt.imsave('./output/ara2013gftt_plant%d.png' % num, img)



    """
    for i in range(1,166):
        if i in range(1,10):
            num = i
            img = cv2.imread('../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_plant00%d_rgb.png' % i)
        if i in range(10,100):
            num = i
            img = cv2.imread('../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_plant0%d_rgb.png' % i)
        if i in range(100,166):
            num = i
            img = cv2.imread('../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_plant00%d_rgb.png' % i)
    print img.format
    plt.imshow(img), plt.show()
    """

    """
    #import images 1-5
    img = cv2.imread('../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_plant001_rgb.png')
    img2 = cv2.imread('../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_plant002_rgb.png')
    img3 = cv2.imread('../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_plant003_rgb.png')
    img4 = cv2.imread('../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_plant004_rgb.png')
    img5 = cv2.imread('../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_plant005_rgb.png')

    # set to gray scale, because OpenCV wants one channel input
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

    # make gausian blurr
    #gray = gD(gray, 3, 1, 1)
    #gray2 = gD(gray2, 3, 1,1)
    #gray3 = gD(gray3, 3, 1,1)
    #gray4 = gD(gray4, 3,1,1)
    #gray5 = gD(gray5, 3,1,1)


    # set to gray scale, because OpenCV wants one channel input
    #gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    #gray2 = cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)
    #gray3 = cv2.cvtColor(gray3, cv2.COLOR_BGR2GRAY)
    #gray4 = cv2.cvtColor(gray4, cv2.COLOR_BGR2GRAY)
    #gray5 = cv2.cvtColor(gray5, cv2.COLOR_BGR2GRAY)


    # maxcorners = 25, quality = 0.01, euclidean dist = 10
    coners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    coners = np.int0(coners)
    corners = cv2.goodFeaturesToTrack(gray2, 25, 0.01, 10)
    corners = np.int0(corners)
    corners3 = cv2.goodFeaturesToTrack(gray3, 25, 0.01, 10)
    corners3 = np.int0(corners3)
    corners4 = cv2.goodFeaturesToTrack(gray4, 25, 0.01, 10)
    corners4 = np.int0(corners4)
    corners5 = cv2.goodFeaturesToTrack(gray5, 25, 0.01, 10)
    corners5 = np.int0(corners5)

    for i in coners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img2, (x,y), 3, 255, -1)

    for i in corners3:
        x,y = i.ravel()
        cv2.circle(img3, (x,y), 3, 255, -1)

    for i in corners4:
        x,y = i.ravel()
        cv2.circle(img4, (x,y), 3, 255, -1)

    for i in corners5:
        x,y = i.ravel()
        cv2.circle(img5, (x,y), 3, 255, -1)


    fig.add_subplot(1,5,1).imshow(img)
    fig.add_subplot(1,5,2).imshow(img2)
    fig.add_subplot(1,5,3).imshow(img3)
    fig.add_subplot(1,5,4).imshow(img4)
    fig.add_subplot(1,5,5).imshow(img5)

    plt.show(fig)
    """
    #cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
