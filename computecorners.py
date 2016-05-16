# vector approach 8 mei 2016


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
import math

tV1 = np.array([[0,1],[-1,0]]).reshape(2,2)
tV2 = np.array([[0,-1], [1,0]]).reshape(2,2)
tV0 = np.array([[-1,0],[0,-1]]).reshape(2,2)

def makeVector(point1, point2):
    [y1, x1] = point1
    [y2, x2] = point2
    newy = y2 - y1
    newx = x2 - x1
    vector = [newy,newx]
    vector = np.array(vector).reshape(1,2)
    return vector

def findCornerPoints(point, vector):
    global tV1, tV2, tV0
    v = np.dot(vector, tV0)
    p0 = point + v
    p1 = p0 + (np.dot(vector, tV1))
    p2 = p0 + (np.dot(vector, tV2))
    return np.array([p1,p2]).reshape(2,2)

def getCorners(point1, point2):
    vector = makeVector(point1, point2)
    halfVector = vector * 0.5
    cornersPoint1 = findCornerPoints(point1, halfVector)
    negvector = makeVector(point2, point1)
    negHalfVector = negvector * 0.5
    cornersPoint2 = findCornerPoints(point2, negHalfVector)
    allCorners = np.append(cornersPoint1, cornersPoint2, axis=0)
    return allCorners

def makePairs(indices):
    pairs = []
    amount = len(indices)
    for i in range(0,amount):
        for j in range(0,amount):
            pointIndex1 = indices[i]
            pointIndex2 = indices[j]
            if not (np.array_equal(pointIndex1,pointIndex2)):
                corners = getCorners(pointIndex1,pointIndex2)
                pairs.append(corners)
    return pairs

# use affine transformation to transform the image
def affineTransform(image, x1, y1, x2, y2, x3, y3, M, N):
    """
    affineTransform
    image(ndarray)  : original image from which an area is used
    x1-3(float)     : x coordinates of 3 corners of the image
    y1-3(float)     : y "
    M(int)          : width of new image
    N(int)          : length of new image

    returns ndarray with shape M, N

    Transforms the area given by the coordinates to the image using affine
    transformation
    """
    # (0,0,M,0,M,N)
    b = np.array([0, 0, M, 0, M, N]).reshape(6, 1)
    A = np.array([[x1, y1, 1, 0, 0, 0], [0, 0, 0, x1, y1, 1],
                  [x2, y2, 1, 0, 0, 0], [0, 0, 0, x2, y2, 1],
                  [x3, y3, 1, 0, 0, 0], [0, 0, 0, x3, y3, 1]])
    v = np.linalg.lstsq(A, b)[0]
    v = v.reshape(2, 3)
    output = cv2.warpAffine(image, v, (M, N))
    return output


################################
"""
def makeVector(point1, point2):
    [[y1], [x1]] = point1
    [[y2], [x2]] = point2
    newy = y2 - y1
    newx = x2 - x1
    vector = [newy, newx]
    vector = np.array(vector).reshape(1,2)
    return vector

def findCornerPoints(point, vector):
    #global tV1, tV2, tV3, tV4
    #print (vector * tV1)
    #a = point + (vector * tV1)
    #b = point + (vector * tV2)
    #c = point + (vector * tV3)
    #d = point + (vector * tV4)

    global tV1, tV2, tV0
    v = np.dot(vector, tV0)
    print v
    p0 = point + np.transpose(v)
    print p0
    v1 = np.dot(vector, tV1)
    v2 = np.dot(vector, tV2)
    p1 = p0 + np.transpose(v1)
    p2 = p0 + np.transpose(v2)
    print p1
    print p2

    return [p1, p2]

    #return [a, b, c, d]

def selectCorners(firstCorners, secondCorners):
    print firstCorners
    firstCorners = np.array(firstCorners).reshape(2,2)
    secondCorners = np.array(secondCorners).reshape(2,2)
    allCorners = np.append(secondCorners, firstCorners, axis=0)
    equalindex = []
    for i in range(0, len(allCorners)-1):
        for j in range(0, len(allCorners)):
            if i != j:
                v = allCorners[i]
                w = allCorners[j]
                if np.array_equal(v,w):
                    equalindex.append(j)
    #allCorners = np.delete(allCorners, equalindex, axis=0)
    print allCorners
    return allCorners



def getCorners(point1, point2):
    global tV0
    point1 = np.array(point1).reshape(2,1)
    point2 = np.array(point2).reshape(2,1)
    vector = makeVector(point1, point2)
    halfVector = vector * 0.5
    cornersPoint1 = findCornerPoints(point1, halfVector)
    negHalfVector = 0.5 * (makeVector(point2, point1))
    cornersPoint2 = findCornerPoints(point2, negHalfVector)
    return selectCorners(cornersPoint1, cornersPoint2)
"""

if __name__ == '__main__':
    test = np.zeros((15,15))
    print test
    plt.imshow(test)
    plt.show()
    point1 = [[4],[4]]
    point2 = [[4],[6]]
    test[4][4] = 1
    test[4][6] = 1
    print test
    plt.imshow(test)
    plt.show()
    corners = getCorners(point1, point2)
    for c in corners:
        [y,x] = c
        test[y][x] = 2
    plt.imshow(test)
    plt.show()
    """
    vector = makeVector(point1, point2)
    print vector
    halfVector = vector * 0.5
    print halfVector
    #point1 = np.array(point1).reshape(2,1)
    a, b, c, d = findCornerPoints(point1, halfVector)
    print a
    print b
    print c
    print d
    [[y1], [x1]] = a
    [[y2], [x2]] = b
    [[y3], [x3]] = c
    [[y4], [x4]] = d
    test[y1][x1] = 2
    test[y2][x2] = 2
    test[y3][x3] = 2
    test[y4][x4] = 2
    plt.imshow(test)
    plt.show()
    print test
    vector = makeVector(point2, point1)
    halfVector = vector * 0.5
    e, f, g, h = findCornerPoints(point2, halfVector)
    print e
    print f
    print g
    print h
    [[y5], [x5]] = e
    [[y6], [x6]] = f
    [[y7], [x7]] = g
    [[y8], [x8]] = h
    test[y5][x5] = 3
    test[y6][x6] = 3
    test[y7][x7] = 3
    test[y8][x8] = 3
    plt.imshow(test)
    plt.show()
    print test
    """
