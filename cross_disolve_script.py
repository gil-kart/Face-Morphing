from cross_disolve_functions import *

path_image = r'imagesAndPoints\face5.tif'
face1 = cv2.imread(path_image)
face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)

path_image = r'imagesAndPoints\face6.tif'
face2 = cv2.imread(path_image)
face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)


imagePts1 = np.load(r"imagesAndPoints\points_in_face_1.npy")
imagePts2 = np.load(r"imagesAndPoints\points_in_face_2.npy")

t_list = np.linspace(0, 1, 130)
im_list = createMorphSequence(face1_gray, imagePts1, face2_gray, imagePts2, t_list, True)

writeMorphingVideo(im_list, 'Example_Video')
