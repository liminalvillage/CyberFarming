
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm



def setHoG(img):
    height = img.shape[0]
    width = img.shape[1]
    cell_size = (width/8, width/8) # 8,8
    block_size = (width/4, width/4) # 16,16
    block_stride = (width/8,width/8) # 8, 8
    nbins = 9
    hog = cv2.HOGDescriptor(_winSize=(width,height),
                            _blockSize=(block_size[1],block_size[0]),
                            _blockStride=(block_stride[1],block_stride[0]),
                            _cellSize=(cell_size[1],cell_size[0]),
                            _nbins=nbins, _winSigma=-1)
    return hog

def getParameters(img, hog):
    width = img.shape[1]
    parameters = hog.compute(img, (width/8,width/8), (0,0))
    return parameters
    """
    height = img.shape[0]
    width = img.shape[1]
    cell_size = (width/8, width/8) # 8,8
    block_size = (width/4, width/4) # 16,16
    block_stride = (width/8,width/8) # 8, 8
    nbins = 9

    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    par = hog.compute(img)
    print par
    print par.shape
    print n_cells
    parameters = hog.compute(img)\
            .reshape(n_cells[1] - block_size[1] + 1,
             n_cells[0] - block_size[0] + 1,
             block_size[0], block_size[1], nbins) \
             .transpose((1, 0, 2, 3, 4))
    print parameters
    #return parameters
    """




def showHoG(img, hog_feats):
    height = img.shape[0]
    width = img.shape[1]
    cell_size = (width/8, width/8) # 8,8
    block_size = (width/4, width/4) # 16,16
    block_stride = (width/8,width/8) # 8, 8
    nbins = 9

    cellsXDir = width/cell_size[0]
    cellsYDir = height/cell_size[0]
    totalnCells = cellsXDir * cellsYDir
    blocksXDir = cellsXDir-1
    blocksYDir = cellsYDir-1
    gradientStrenghts = np.zeros((cellsYDir,cellsXDir, nbins))
    cellUpdateCOunter = np.zeros((cellsYDir, cellsXDir))
    index = 0
    for x in range(blocksXDir):
        for y in range(blocksYDir):
            for c in range(0,4):
                cellx = x
                celly = y
                if c == 1:
                    celly += 1
                if c == 2:
                    cellx += 1
                if c == 3:
                    celly += 1
                    cellx += 1
                for b in range(0,nbins):
                    gs = hog_feats[index]
                    index += 1
                    gradientStrenghts[celly][cellx][b] += gs
                cellUpdateCOunter[celly][cellx] += 1

    for cellx in range(cellsXDir):
        for celly in range(cellsYDir):
            updates = cellUpdateCOunter[celly][cellx]
            for b in range(0,nbins):
                thatBin = gradientStrenghts[celly][cellx][b]
                print "before average", thatBin
                gradientStrenghts[celly][cellx][b] = thatBin/updates
                print "after average", gradientStrenghts[celly][cellx][b]


    print gradientStrenghts
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    #ax1.axis('off')
    #ax1.imshow(img, cmap=cm.Greys)
    #ax1.set_title('Input img')
    #ax1.set_adjustable('box-forced')
    #ax2.axis('off')
    #ax2.imshow(gradientStrenghts[:,:,:-1])
    #ax2.set_title('hog')
    #ax2.set_adjustable('box-forced')
    #plt.show()

    plt.imshow(img, cmap=cm.gray)
    plt.show()

    bin = 5
    plt.pcolor(gradientStrenghts[:,:,bin])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    #plt.figure()
    #plt.imshow(img, cmap='gray')
    plt.show()






    """
    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    gradients = np.zeros((n_cells[0], n_cells[1], nbins))

    # count cells (border cells appear less often across overlapping groups)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                      off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                hog_feats[:, :, off_y, off_x, :]
            cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                       off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

    # Average gradients
    gradients /= cell_count

    # Preview
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

    bin = 5  # angle is 360 / nbins * direction
    plt.pcolor(gradients[:, :, bin])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.show()
    """
