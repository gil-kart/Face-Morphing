import cv2.cv2
import numpy as np

from hw2_functions import *

# ----------------------------- question a --------------------------------------

path_image = r'FaceImages\face5.tif'
face1 = cv2.imread(path_image)
face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)

path_image = r'FaceImages\face6.tif'
face2 = cv2.imread(path_image)
face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)


imagePts1 = np.load("points_in_face_1.npy")
imagePts2 = np.load("points_in_face_2.npy")



t_list = np.linspace(0, 1, 130)
im_list = createMorphSequence(face1_gray, imagePts1, face2_gray, imagePts2, t_list, True)

writeMorphingVideo(im_list, 'Example_Video')

# --------------------- question B -------------------------

path_image = r'FaceImages\target.tif'
target = cv2.imread(path_image)
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

path_image = r'FaceImages\sqrtar.tif'
tri = cv2.imread(path_image)
tri_gray = cv2.cvtColor(tri, cv2.COLOR_BGR2GRAY)



imagePts1 = np.load("tri_1.npy")
imagePts2 = np.load("tri_2.npy")


affin = findAffineTransform(imagePts1, imagePts2)
proj = findProjectiveTransform(imagePts1, imagePts2)


proj_ret = mapImage(tri_gray, proj, [tri_gray.shape[0], tri_gray.shape[1]])
affin_ret = mapImage(tri_gray, affin, [tri_gray.shape[0], tri_gray.shape[1]])
print("the 4 source points, are creating the black rectangle\n")
print("the target picture is supposed to look like this after the transformation\n")
plt.imshow(target_gray, cmap='gray')

plt.figure()

plt.subplot(1, 4, 1)
plt.imshow(target_gray, cmap='gray')
plt.title("transformation goal")
plt.subplot(1, 4, 2)
plt.imshow(proj_ret, cmap='gray')
plt.title("projective")
plt.subplot(1, 4, 3)
plt.imshow(affin_ret, cmap='gray')
plt.title("affin")
plt.subplot(1, 4, 4)
plt.imshow(tri_gray, cmap='gray')
plt.title("origin")

# ----------------------- question C --------------------------------
# ---------------------------- i ------------------------------------
path_image = r'FaceImages\face6.tif'
im1_color = cv2.imread(path_image)
im1 = cv2.cvtColor(im1_color, cv2.COLOR_BGR2GRAY)

path_image = r'FaceImages\face4.tif'
im2_color = cv2.imread(path_image)
im2 = cv2.cvtColor(im2_color, cv2.COLOR_BGR2GRAY)

im1_pts = np.load("q3_im1_pts.npy")
im2_pts = np.load("q3_im2_pts.npy")
im1_pts_4pts = np.load("q3_im1_pts_4pts.npy")
im2_pts_4pts = np.load("q3_im2_pts_4pts.npy")

t = 0.5
one_2_two_12pts = findProjectiveTransform(im1_pts, im2_pts)
two_2_one_12pts = findProjectiveTransform(im2_pts, im1_pts)
one_2_two_4pts = findProjectiveTransform(im1_pts_4pts, im2_pts_4pts)
two_2_one_4pts = findProjectiveTransform(im2_pts_4pts, im1_pts_4pts)


# # ------------------ for image with 4 points ---------------
T12_t = (1 - t) * np.eye(3) + t * one_2_two_4pts
T21_t = (1 - t) * two_2_one_4pts + t * np.eye(3)
new_im_1 = mapImage(im1, T12_t, [im1.shape[0], im1.shape[1]])
new_im_2 = mapImage(im2, T21_t, [im1.shape[0], im1.shape[1]])
nim_4pts = ((1-t)*new_im_1 + t * new_im_2).astype(np.uint8)

# # ------------------ for image with 12 points -------------
T12_t = (1 - t) * np.eye(3) + t * one_2_two_12pts
T21_t = (1 - t) * two_2_one_12pts + t * np.eye(3)
new_im_1 = mapImage(im1, T12_t, [im1.shape[0], im1.shape[1]])
new_im_2 = mapImage(im2, T21_t, [im1.shape[0], im1.shape[1]])
nim_12pts = ((1-t)*new_im_1 + t * new_im_2).astype(np.uint8)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(nim_4pts, cmap='gray')
plt.title("4 points")
plt.subplot(1, 2, 2)
plt.imshow(nim_12pts, cmap='gray')
plt.title("12 points")


# # ---------------------------- ii ------------------------------------


im1_bad_dis = np.load("points_face_1_bad_dis.npy")
im2_bad_dis = np.load("points_face_2_bad_dis.npy")


one_2_two_bad_dis = findProjectiveTransform(im1_bad_dis, im2_bad_dis)
two_2_one_bad_dis = findProjectiveTransform(im2_bad_dis, im1_bad_dis)
T12_t = (1 - t) * np.eye(3) + t * one_2_two_bad_dis
T21_t = (1 - t) * two_2_one_bad_dis + t * np.eye(3)
new_im_1 = mapImage(im1, T12_t, [im1.shape[0], im1.shape[1]])
new_im_2 = mapImage(im2, T21_t, [im1.shape[0], im1.shape[1]])
nim_bad_dis = ((1-t)*new_im_1 + t * new_im_2).astype(np.uint8)


print("the group of points that are distributed badly are distributed in this way: \n")
print("most of the points are at the top of the persons head, meaning, most of them are at the eyes and hair and a \
small part of the are in the chin and cheeks")
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(nim_12pts, cmap='gray')
plt.title("points distributed well")
plt.subplot(1, 2, 2)
plt.imshow(nim_bad_dis, cmap='gray')
plt.title("points distributed badly")


