import numpy as np
import cv2
import matplotlib.pyplot as plt

def writeMorphingVideo(image_list, video_name):
    out = cv2.VideoWriter(video_name + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, image_list[0].shape, 0)
    for im in image_list:
        out.write(im)
    out.release()

def createMorphSequence(im1, im1_pts, im2, im2_pts, t_list, transformType):
    if transformType:
        trantype_im1_to_im2 = findProjectiveTransform(im1_pts, im2_pts)
        trantype_im2_to_im1 = findProjectiveTransform(im2_pts, im1_pts)
    else:
        trantype_im1_to_im2 = findAffineTransform(im1_pts, im2_pts)
        trantype_im2_to_im1 = findAffineTransform(im2_pts, im1_pts)

    size = [im2.shape[0], im2.shape[1]]
    ims = []
    one_mat = np.eye(3)

    for t in t_list:
        T12_t = (1 - t) * one_mat + t * trantype_im1_to_im2
        T21_t = (1 - t) * trantype_im2_to_im1 + t * one_mat
        new_im_1 = mapImage(im1, T12_t, size)
        new_im_2 = mapImage(im2, T21_t, size)
        nim = ((1-t)*new_im_1 + t * new_im_2).astype(np.uint8)
        #plt.imshow(nim, cmap='gray')
        ims.append(nim)
    return ims


def mapImage(im, T, sizeOutIm):

# create meshgrid of all coordinates in new image [x,y]
    array_1 = np.array([i for i in range(sizeOutIm[0])])
    array_2 = np.array([i for i in range(sizeOutIm[1])])
    yy, xx = np.meshgrid(array_1, array_2)
    array_3 = [1 for i in range(yy.shape[0])]
    zz, zzx = np.meshgrid(array_3, array_1)
    xx = xx.ravel()
    yy = yy.ravel()
    zz = zz.ravel()
    xy = np.vstack([yy, xx, zz])
    Tinv = np.linalg.pinv(T)
    new_mat = np.matmul(Tinv, xy)

    new_mat[0] = new_mat[0] / new_mat[2]
    new_mat[1] = new_mat[1] / new_mat[2]

    yy = new_mat[0]
    xx = new_mat[1]
    xMask = np.logical_or(xx < 0, xx >= im.shape[1] - 1)
    yMask = np.logical_or(yy >= im.shape[0] - 1, yy < 0)
    outOfRange = np.logical_not(np.logical_or(xMask, yMask))
    xx = xx[outOfRange]
    yy = yy[outOfRange]

    yy_, xx_ = np.meshgrid(np.arange(0, sizeOutIm[0]), np.arange(0, sizeOutIm[1]))
    xx_ = xx_.ravel()
    yy_ = yy_.ravel()
    xx_ = xx_[outOfRange]
    yy_ = yy_[outOfRange]

    SEpx = np.ceil([xx, yy]).astype(np.uint8)
    NWpx = np.floor([xx, yy]).astype(np.uint8)
    NEpx = np.array([np.ceil(xx).astype(np.uint8), np.floor(yy).astype(np.uint8)])
    SWpx = np.array([np.floor(xx).astype(np.uint8), np.ceil(yy).astype(np.uint8)])

    SEcolor = np.zeros((sizeOutIm[0], sizeOutIm[1]))
    NWcolor = np.zeros((sizeOutIm[0], sizeOutIm[1]))
    NEcolor = np.zeros((sizeOutIm[0], sizeOutIm[1]))
    SWcolor = np.zeros((sizeOutIm[0], sizeOutIm[1]))

    SEcolor[xx_, yy_] = im[SEpx[0].astype(int), SEpx[1].astype(int)]
    NWcolor[xx_, yy_] = im[NWpx[0].astype(int), NWpx[1].astype(int)]
    NEcolor[xx_, yy_] = im[NEpx[0].astype(int), NEpx[1].astype(int)]
    SWcolor[xx_, yy_] = im[SWpx[0].astype(int), SWpx[1].astype(int)]

    SEpxfull = np.ceil([new_mat[1], new_mat[0]]).astype(np.uint8)

    X_, Y_ = new_mat[1], new_mat[0]
    deltaX = (SEpxfull[0] - X_)
    deltaY = (SEpxfull[1] - Y_)
    deltaX = deltaX.reshape((sizeOutIm[0], sizeOutIm[1]))
    deltaY = deltaY.reshape((sizeOutIm[0], sizeOutIm[1]))

    S = SEcolor * (1 - deltaX) + SWcolor * deltaX
    N = NEcolor * (1 - deltaX) + NWcolor * deltaX
    V = (N * deltaY + S * (1 - deltaY)).astype(np.uint8)
    return V



def findProjectiveTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]
    X = [[]]
    # iterate iver points to create x , x'
    Xtag = np.zeros((2 * N, 1))

    for i in range(0, N):
        array_row_i = np.array([[pointsSet1[i, 0], pointsSet1[i, 1], 0, 0, 1, 0, -(pointsSet1[i, 0] * pointsSet2[i, 0])
                           , -(pointsSet1[i, 1] * pointsSet2[i, 0])]])
        array_row_i_plus_1 = np.array([[0, 0, pointsSet1[i, 0], pointsSet1[i, 1], 0, 1, -(pointsSet1[i, 0] * pointsSet2[i, 1])
                                  , -(pointsSet1[i, 1] * pointsSet2[i, 1])]])
        X = np.append(X, array_row_i)
        X = np.append(X, array_row_i_plus_1)
        Xtag[2 * i], Xtag[2 * i + 1] = pointsSet2[i, 0], pointsSet2[i, 1]
    ####calculate T - be careful of order when reshaping it

    X = np.reshape(X, (2 * N, 8))
    pinv = np.linalg.pinv(X)
    T = np.matmul(pinv, Xtag)

    project = np.zeros((3, 3))
    project[0, 0] = T[0]
    project[0, 1] = T[1]
    project[0, 2] = T[4]
    project[1, 0] = T[2]
    project[1, 1] = T[3]
    project[1, 2] = T[5]
    project[2, 0] = T[6]
    project[2, 1] = T[7]
    project[2, 2] = 1
    return project


def findAffineTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]
    X = [[]]
    # iterate iver points to create x , x'
    Xtag = np.zeros((2 * N, 1))

    for i in range(0, N):
        array_row_i = [[pointsSet1[i, 0], pointsSet1[i, 1], 0, 0, 1, 0]]
        array_row_i_plus_1 = [[0, 0, pointsSet1[i, 0], pointsSet1[i, 1], 0, 1]]
        X = np.append(X, array_row_i)
        X = np.append(X, array_row_i_plus_1)
        Xtag[2 * i], Xtag[2 * i + 1] = pointsSet2[i, 0], pointsSet2[i, 1]
    ####calculate T - be careful of order when reshaping it
    X = np.reshape(X, (2 * N, 6))
    pinv = np.linalg.pinv(X)
    T = np.matmul(pinv, Xtag)

    affine = np.zeros((3, 3))
    affine[0, 0] = T[0]
    affine[0, 1] = T[1]
    affine[0, 2] = T[4]
    affine[1, 0] = T[2]
    affine[1, 1] = T[3]
    affine[1, 2] = T[5]
    affine[2, 2] = 1
    return affine


def getImagePts(im1, im2, varName1, varName2, nPoints):
    plt.imshow(im1, cmap='gray')
    temp_1 = np.round(plt.ginput(nPoints, show_clicks=True,  timeout=300))

    plt.imshow(im2, cmap='gray')
    temp_2 = np.round(plt.ginput(nPoints, show_clicks=True,  timeout=300))



    #imagePts1 = np.append(np.fliplr(temp_1), np.ones((nPoints, 1), dtype=np.uint8), axis=1)
    #imagePts2 = np.append(np.fliplr(temp_2), np.ones((nPoints, 1), dtype=np.uint8), axis=1)
    imagePts1 = np.append(temp_1, np.ones((nPoints, 1), dtype=np.uint8), axis=1)
    imagePts2 = np.append(temp_2, np.ones((nPoints, 1), dtype=np.uint8), axis=1)
    np.save(varName1 + ".npy", imagePts1)
    np.save(varName2 + ".npy", imagePts2)

def createMorphSequence2(im1, im1_pts, t_list, transformType):
    ims = []
    for i in range(0, 361, 5):
        c, s = np.cos(np.deg2rad(i)), np.sin(np.deg2rad(i))
        rMatrix = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        img3 = mapImage(im1, rMatrix, im1.shape)
        ims.append(img3.astype(np.uint8))
    return ims

