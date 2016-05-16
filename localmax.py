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
import math

tV1 = np.array([[0,1],[-1,0]]).reshape(2,2)
tV2 = np.array([[0,-1], [1,0]]).reshape(2,2)
tV0 = np.array([[-1,0],[0,-1]]).reshape(2,2)


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


def getThreshold(values):
    mini = min(values)
    maxi = max(values)
    std = np.std(values)
    return ((maxi-std + mini)/2)


def getMaxWithThreshold(maxima, values):
    threshold = getThreshold(values)
    nonzero = np.transpose(np.nonzero(maxima))
    for i in nonzero:
        if maxima[i[0]][i[1]] < threshold:
            maxima[i[0]][i[1]] = 0
    indices = np.transpose(np.nonzero(maxima))
    return indices


def getLocalMax(G):
    width = G.shape[1]
    hight = G.shape[0]
    values = []
    maxima = np.zeros((hight,width))
    for x in range(0,width-1):
        for y in range(0,hight-1):
            # if local maxima
            if (isMax(G,y,x)):
                maxima[y][x] = G[y][x]
                values.append(G[y][x])
    return getMaxWithThreshold(maxima, values)
    #print maxima
    #print values
    #print np.std(values)
    #med = median(values)
    #print med
    #mean = np.mean(values)
    #std = np.std(values)
    #meanstd = std/len(values)
    #print mean
    #if mean < std:
    #    threshold = mean - meanstd
    #else:
    #    threshold = std - meanstd
    #threshold = np.mean(values)
    #nonzero = np.transpose(np.nonzero(maxima))
    #for i in nonzero:
    #    if maxima[i[0]][i[1]] < threshold:
    #        maxima[i[0]][i[1]] = 0
    #indices = np.transpose(np.nonzero(maxima))

    # indices of the points to extract features
    #nonzero = np.transpose(np.nonzero(maxima))
    #plt.imshow(maxima, cmap=cm.Greys)
    #plt.show()
    #print len(indices)
    #return indices

"""
def assignPointsForTransformation(img, coord1, coord2):
    dist = np.linalg.norm(coord1- coord2)


def manualRange(start, stop, step):
    mrange = []
    while start < stop:
        mrange.append(float(start))
        start += step
    return mrange

def pointsInRange(r, cx, cy, s=0.1,n=100):
    mrange = manualRange(0,n,s)
    points = []
    for i in mrange:
        xpoint = cx + math.cos(2*pi/n*i)*r
        ypoint = cy + math.sin(2*pi/n*i)*r
        if xpoint >= 0 and ypoint >= 0:
            points.append([ypoint,xpoint])
    return points

def getThosePoints(dist, point1, point2, factor):
    [cy1, cx1] = point1
    [cy2, cx2] = point2
    # flag = the requited distance factor
    intermediates = pointsInRange((dist/2), cx1, cy1)
    # with pitagoras I got the following numbers for the relative distance
    requiredDist = pythagoras(dist, factor)
    if factor == 1:
        half = goodDistPoint(intermediates, point2, requiredDist, dist, 2)
    else:
        half = goodDistPoint(intermediates, point2, requiredDist, dist, 1)
    return half

def cornerCoord(point1, point2):
    [cy1, cx1] = point1
    [cy2, cx2] = point2
    corners = []
    dist = np.linalg.norm(point1 - point2)
    noCorners = getThosePoints(dist, point1, point2, 1)
    for no in noCorners:
        corner = getThosePoints(dist, no, point2, 1.5)
        corners.append(corner)
    return corners

    dist = np.linalg.norm(point1 - point2)
    # get points to find the first corners
    intermediates = pointsInRange((dist/2), cx1, cy1)
    # with pitagoras I got the following numbers for the relative distance
    requiredDist = pythagoras(dist, 1)
    half = goodDistPoint(intermediates, point2, requiredDist, 2)
    for h in half:
        intermediates = pointsInRange((dist/2), h[1], h[0])
        requiredDist = pythagoras(dist, 1.5)
        corner = goodDistPoint(intermediates, point2, requiredDist, 1)
        corners.append(corner)
    print corners
    return corners


def pythagoras(dist, factor):
    a = dist * factor
    b = dist * .5
    return math.sqrt((a*a)+(b*b))

def goodDistPoint(possible, point2, dist, wholedist, amount):
    halfs = []
    for point in possible:
        distance = np.linalg.norm(point - point2)
        halfs.append([abs(dist-distance), point])
    best = sorted(halfs)
    if amount == 1:
        toreturn = best[0][1]
    else:
        for i in range(1, len(best)):
            print best[0][1]
            print best[i][1]
            d = np.linalg.norm(np.array(best[0][1]) - np.array(best[i][1]))
            if d > (wholedist/2):
                toreturn = [best[0][1], best[i][1]]
                break
    return toreturn


def firstMark(point1, point2):
    mark1 = []
    mark2 = []
    dist = np.linalg.norm(point1 - point2)
    possibleMark1 = pointsInRange(dist/2, point1[1], point1[0])
    possibleMark2 = pointsInRange(dist/2, point2[1], point2[0])
    toReach = dist * 1.5
    for mark in possibleMark1:
        distance = np.linalg.norm(mark - point2)
        if toReach == distance:
            mark1.append(mark)
    for mark in possibleMark2:
        distance = np.linalg.norm(mark - point1)
        if toReach == distance:
            mark2.append(mark)
    print mark1
    print mark2
    return mark1, mark2

    """
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
    b = np.array([0, 0, M, 0, M, N]).reshape(6, 1)
    A = np.array([[x1, y1, 1, 0, 0, 0], [0, 0, 0, x1, y1, 1],
                  [x2, y2, 1, 0, 0, 0], [0, 0, 0, x2, y2, 1],
                  [x3, y3, 1, 0, 0, 0], [0, 0, 0, x3, y3, 1]])
    v = np.linalg.lstsq(A, b)[0]
    v = v.reshape(2, 3)
    output = cv2.warpAffine(image, v, (M, N))
    return output



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

def convertIndices(indices, bordersize):
    toAdd = np.array([bordersize, bordersize])
    newIndices = np.array([])
    for i in range(0, len(indices)):
        new = indices[i] + toAdd
        if i == 0:
            newIndices = new
        else:
            newIndices = np.vstack((newIndices,new))
    return newIndices



def extractFeatures(image, indices):
    bs = 200
    borderReplicate = cv2.copyMakeBorder(image, bs,bs,bs,bs,cv2.BORDER_REPLICATE)
    plt.imshow(borderReplicate,cmap=cm.gray)
    plt.show()
    convertIndices(indices, bs)
    print len(indices)
    pairs = makePairs(indices)
    print len(pairs)
    counter = 0
    notpossible = 0
    for pair in pairs[7]:
        possible = True
        # check if affine transformation is possible
        for coordinates in pair:
            if coordinates[0] < 0 or coordinates[1] < 0:
                notpossible = notpossible + 1
        #        #print "No feature Possible"
                possible = False
        if possible:
            #print "not possibles ", notpossible
            counter = counter + 1
            yaxis = pair[:, :-1]
            xaxis = pair[:, -1:]
            print pair
            extractedArea = affineTransform(image, pair[0][1],pair[0][0],pair[1][1],pair[1][0],pair[2][1],pair[2][0], 200,100)
            #print counter
            #print extractedArea
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            cn = 0
            for c in pair:
                cn = cn + 1
                [y,x] = c
                y = int(y)
                x = int(x)
                cv2.circle(image, (x,y), 1, cn, -1)
            for i in indices:
                y, x = i.ravel()
                cv2.circle(image, (x,y), 3, 0, -1)
            ax1.axis('on')
            ax1.imshow(image, cmap=cm.Greys)
            ax1.set_title('Input img')
            ax1.set_adjustable('box-forced')
            ax2.axis('on')
            ax2.imshow(extractedArea, cmap=cm.Greys)
            ax2.set_title('affine ')
            ax2.set_adjustable('box-forced')
            #plt.show()
    print counter






if __name__=="__main__":
    mypath = './output/saliency'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n]= cv2.imread(join(mypath, onlyfiles[n]))

    for i in range(4, 10):
        print "Image ", i
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
        for j in range(0,len(pointIndices)):
            k = pointIndices[j]
            y,x = k.ravel()
            cv2.circle(images[i], (x,y), 1, 0, -1)
        plt.imshow(images[i], cmap=cm.Greys)
        plt.show()
        extractFeatures(images[i], pointIndices)
        #corners = getCorners(pointIndices[7], pointIndices[8])
        #for c in corners:
        #    [y,x] = c
        #    y = int(y)
        #    x = int(x)
        #    cv2.circle(images[i], (x,y), 2, 0, -1)
        #m1, m2 = firstMark(pointIndices[7], pointIndices[8])
        #[c1, c2] = cornerCoord(pointIndices[0], pointIndices[1])
        #[c3, c4] = cornerCoord(pointIndices[1], pointIndices[0])
        #coordinates = [c1,c2,c3,c4]

        #for j in range(7,9):
        #    k = pointIndices[j]
        #    y,x = k.ravel()
        #    cv2.circle(images[i], (x,y), 3, 255, -1)
        #if len(m1) > 0:
        #    for m in range(0, len(m1)):
        #        k = m1[m]
        #        y = int(k[0])
        #        x = int(k[1])
        #        cv2.circle(images[i], (x,y), 2, 0, -1)
        #if len(m2) > 0:
        #    for mm in range(0, len(m2)):
        #        l = m2[mm]
        #        y = int(l[0])
        #        x = int(l[1])
        #        cv2.circle(images[i], (x, y), 1, 100, -1)
        plt.imshow(img, cmap=cm.Greys)
        plt.show()
        """
        for co in coordinates:
            y = int(co[0])
            x = int(co[1])
            #y,x = c.ravel()
            cv2.circle(images[i], (x,y), 2, 0, -1)
        plt.imshow(img, cmap=cm.Greys)
        plt.show()
        """




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
