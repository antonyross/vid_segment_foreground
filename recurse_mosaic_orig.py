# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 02:08:35 2014

@author: antonyr
"""

import cv2, numpy as np


def extract_features_and_descriptors(image):

  ## Convert image to grayscale (for SURF detector).
  gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
  
  ## Detect SURF features and compute descriptors.
  keypoints = []
  descriptors = []
  
  keypoints = detect_features(gray_image)
  descriptors = extract_descriptors(gray_image, keypoints)
  
  return (keypoints, descriptors)
  
## --------------------------------------------------------------------
def detect_features(grey_image):
  surf = cv2.FeatureDetector_create("SURF")
  surf.setDouble("hessianThreshold", 1000)
  return surf.detect(grey_image)
  
def extract_descriptors(grey_image, keypoints):
  surf = cv2.DescriptorExtractor_create("SURF")
  return surf.compute(grey_image, keypoints)[1]
  

## --------------------------------------------------------------------
## 3.2 Find corresponding features between the images. ----------------
def find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2):

  ## Find corresponding features.
  match = match_flann(descriptors1, descriptors2)
  
  
  ## Look up corresponding keypoints.
  points1 = []
  points2 = []

  # get the corresponding points from the respective index in the match tuple
  for m in match:
    i, j = m
    points1.append(keypoints1[i].pt)
    points2.append(keypoints2[j].pt)


  return (points1, points2)



## 3.4  Combine images into a panorama. --------------------------------
def merge_images(image1, image2, homography, size, offset, keypoints):

  
  #image2 = cv2.warpPerspective(image2, homography, size)  
  ## Combine the two images into one.

  image2 = cv2.warpPerspective(image2,homography,size)
  print image1.shape, 'image1 shape'
  print image2.shape, 'image2 shape'
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  panorama = np.zeros((max(size[1], h1), max(size[0], w1), 3), np.uint8) # max is used to choose the max ht between image 2 (size[1]) and the panorama/image1 (h1)
  
  print image2.shape, "shape"
  panorama[:h2, :w2] = image2
  panorama[:h1, :w1] = image1
  #panorama[:h2, -90:] = image2[:, -90:]

  #panorama = np.zeros((size[1], size[0], 3), np.uint8)  
  
  #place_image(panorama, image2, offset[0], offset[1])

  return panorama






def match_flann(desc1, desc2, r_threshold = 0.5):
  # 'Finds strong corresponding features in the two given vectors.'
  
  if len(desc1) == 0 or len(desc2) == 0:
    print "No features passed into match_flann"
    return []

  ## Build a kd-tree from the second feature vector.
  FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
  flann = cv2.flann_Index(desc2, {'algorithm': FLANN_INDEX_KDTREE, 'trees': 4})

  ## For each feature in desc1, find the two closest ones in desc2.
  (idx2, dist) = flann.knnSearch(desc1, 2, params={}) # bug: need empty {}

  ## Create a mask that indicates if the first-found item is sufficiently
  ## closer than the second-found, to check if the match is robust.
  mask = dist[:,0] / dist[:,1] < r_threshold
  
  ## Only return robust feature pairs.
  idx1  = np.arange(len(desc1))
  pairs = np.int32(zip(idx1, idx2[:,0]))
  return [(i,j) for (i,j) in pairs[mask]]


def draw_correspondences(image1, image2, points1, points2):
  'Connects corresponding features in the two images using yellow lines.'

  ## Put images side-by-side into 'image'.
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
  image[:h1, :w1] = image1
  image[:h2, w1:w1+w2] = image2
  
  ## Draw yellow lines connecting corresponding features.
  for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
    cv2.line(image, (x1, y1), (x2+w1, y2), (0, 255, 255), lineType=cv2.CV_AA)

  return image




## ---------------------------------------------------------------------

def show(name, im):
  if im.dtype == np.complex128:
    raise Exception("OpenCV can't operate on complex valued images")
  cv2.namedWindow(name)
  cv2.imshow(name, im)
  cv2.waitKey(1)
  
  
def getHomography(img1, img2):
  ## Load images.
  image1 = img1
  image2 = img2

  ## Detect features and compute descriptors.
  keypoints1, descriptors1 = extract_features_and_descriptors(image1)
  keypoints2, descriptors2 = extract_features_and_descriptors(image2)
  
  #show("Image1 features", cv2.drawKeypoints(image1, keypoints1, color=(0,0,255)))
  #show("Image2 features", cv2.drawKeypoints(image2, keypoints2, color=(0,0,255)))
  
  ## Find corresponding features.
  points1, points2 = find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2)
  points1 = np.array(points1, dtype=float)
  points2 = np.array(points2, dtype=float)
  #print len(points1), "features matched"

  ## Visualise corresponding features.
  #correspondences = draw_correspondences(image1, image2, points1, points2)
  #cv2.imwrite("correspondences.jpg", correspondences)
  #cv2.imshow('correspondences', correspondences)
  
  ## Find homography between the views.
  if len(points1) < 4 or len(points2) < 4:
    print "Not enough features to find a homography"
    homography = np.identity(3, dtype=float)
  else:
    (homography, _) = cv2.findHomography(points2, points1, method=cv2.RANSAC)
    #homography = cv2.estimateRigidTransform( img2,  img1, False)
    #homography = np.matrix(homography)

  
  
  ## Calculate size and offset of merged panorama.
#==============================================================================
#   (size, offset) = calculate_size(reference_frame_size, image2.shape[:2], homography)
#   size = tuple(np.asarray(size).flatten().astype(int).tolist())
#   offset = tuple(np.asarray(offset).flatten().astype(int).tolist())
#   print "output size: %ix%i" % size
#==============================================================================
  
  #size  = (726, 360) 
  #offset = (86.67180289,   0.21873984)
  
  ## Finally combine images into a panorama.
  #pano = merge_images(image1, image2, homography, size, offset, (points1, points2))
  
  return homography
  
 
def getHomographiesToReferenceFrame(interFrame_homographies, homogs_to_reference_frame):
  #n = len(interFrame_homographies)
  for h in interFrame_homographies:
    homogs_to_reference_frame.append(np.dot(homogs_to_reference_frame[-1], h))
  return homogs_to_reference_frame
  


def calculate_size(next_img_size, H):
  #Compute size of output panorama image
  min_row = 0
  min_col = 0
  max_row = 0
  max_col = 0

  rows, columns = next_img_size
    
  #create a matrix with the coordinates of the four corners of the current image
  corner_points_matrix = np.array([[0, 0, 1], [0,rows,1], [columns, 0, 1], [columns, rows, 1]])  # points:  top left, bottom left, top right, bottom right
    
  #Map each of the 4 corner's coordinates into the coordinate system of the reference image
  
  
  
  point_locations_array = []
  for p in corner_points_matrix:
    point_location_vector = np.dot(H, p)  # homogeneous coordinates ... [x,y,w]
    point_location_vector = point_location_vector/point_location_vector[2]  # divide by w to get inhomogeneous vector [x,y,1]
    
    #min_row = min(min_row, int(point_location_vector[1])) # compare y value
    #min_col = min(min_col, result(2))  
    #max_row = max(max_row, int(point_location_vector[1])) # compare y value
    #max_col = max(max_col, int(point_location_vector[0])) # compare x value
    
    point_locations_array.append(point_location_vector)  # top left, bottom left, top right, bottom right

  
  # Calculate output image size
  #image_columns = max_col - min_col
  #image_rows = max_row - min_row
  
  
  image_columns = min(int(point_locations_array[2][0]), int(point_locations_array[3][0])) # get the minimum of the top/bottom right point widths to crop away black border
  image_rows = max(int(point_locations_array[1][1]), int(point_locations_array[3][1]))
  

  #Calculate offset of the upper-left corner of the reference image relative to the upper-left corner of the output image
  #row_offset = ...
  #col_offset = ...
  #offset = (col_offset, row_offset)
  
  ## Update the homography to shift by the offset
  #homography[0,2] += offset[0]   i.e. location_vector[0.2] += offset[0]
  #homography[1,2] += offset[1]   i.e. location_vector[1.2] += offset[1]
  
  size = (image_columns, image_rows)
  
  return size
  
  
  
def place_image(mosaic, next_image, homography, size, padding):
  #panorama = np.hstack(panorama, np.zeros((panorama.shape[0],padding,3), np.uint8))
  size = (size[0] + padding, size[1])                 
                    #np.zeros((panorama.shape[0],panorama.shape[1]+padding,panorama.shape[2]))  
  #horiz_size = size + padding
  next_image = cv2.warpPerspective(next_image, homography, size)
  #panorama = cv2.warpAffine(image, homography[:2], (1000, 368))
  
  (h1, w1) = mosaic.shape[:2]
  (h2, w2) = next_image.shape[:2]
  #panorama = np.zeros((max(size[1], h1), max(size[0], w1), 3), np.uint8) # max is used to choose the max ht between image 2 (size[1]) and the panorama/image1 (h1)
  panorama = np.zeros((size[1], size[0], 3), np.uint8) # max is used to choose the max ht between image 2 (size[1]) and the panorama/image1 (h1)

  panorama[:h1, :w1] = mosaic
  panorama[:h2, -150:] = next_image[:, -150:]
  

  return panorama, size    


def place_image2(mosaic, next_image, homography): 
  
  size = calculate_size(next_image.shape[:2], homography)             
                  
  panorama = cv2.warpPerspective(next_image, homography, size)
  #panorama = cv2.warpAffine(image, homography[:2], (1000, 368))
  
  (h1, w1) = mosaic.shape[:2]  # mosaic is the previous panorama
  (h2, w2) = panorama.shape[:2]  # panorama is the new composition
  
  panorama[:h1, :w1] = mosaic[:min(h1, h2), :min (w1,w2)]
  
  return panorama    
  
  
  
  
def place_image3(mosaic, next_image, homography, far_left_offset): 
  
  offset = homography[2,2] * far_left_offset
  homography[0,2] += offset
  

  
  size = calculate_size(next_image.shape[:2], homography) 
             
                  
  panorama = cv2.warpPerspective(next_image, homography, size)
  #panorama = cv2.warpAffine(image, homography[:2], (1000, 368))
  
  (h1, w1) = mosaic.shape[:2]  # mosaic is the previous panorama
  (h2, w2) = panorama.shape[:2]  # panorama is the new composition
  
  panorama[:h1, :w1] = mosaic[:min(h1, h2), :min (w1,w2)]
  
  return panorama 
  

      

def place_image_once(ref_image, next_image, homography, size):
    
  # --> size = calculate_size(next_image.shape[:2], homography)
  panorama = cv2.warpPerspective(next_image, homography, size)
  #panorama = cv2.warpAffine(image, homography[:2], (1000, 368))
  
  (h1, w1) = ref_image.shape[:2]
  
  #panorama = np.zeros((max(size[1], h1), max(size[0], w1), 3), np.uint8) # max is used to choose the max ht between image 2 (size[1]) and the panorama/image1 (h1)
  #panorama = np.zeros((size[1], size[0], 3), np.uint8) # max is used to choose the max ht between image 2 (size[1]) and the panorama/image1 (h1)

  #print image2.shape, "shape"
  #panorama[:h2, :w2] = image2
  # --> panorama[:h1, :w1] = ref_image
  #panorama[:h2, -150:] = next_image[:, -150:]
  

  return panorama   
      

  
      
def homographies_to_first_frame(folder, initial_to_frame, initial_from_frame, frame_increment, total_number_of_frames):
  i = initial_from_frame # the initial from_frame number
  
  to_frm = 'frame' + str(initial_to_frame) + '.jpg'
  to_frm = folder + to_frm
  
  to_frame = cv2.imread(to_frm)
  
  interFrame_homographies = []
  homographies_to_reference_frame = [np.array([[1,0,0], [0,1,0], [0,0,1]])]
  frames_to_transform = [to_frm] # initial to_frame is at index 0
  
  while(i <= total_number_of_frames):    
    next_jpg = 'frame' + str(i) + '.jpg'
    next_jpg = folder + next_jpg
    from_frame = cv2.imread(next_jpg)
    interFrame_homographies.append(getHomography(to_frame, from_frame))
     
    frames_to_transform.append(next_jpg)
    to_frame = from_frame[:] # places a copy of from_frame into to_frame
    
    i += frame_increment  # increment size
  
  
  homographies_to_reference_frame = getHomographiesToReferenceFrame(interFrame_homographies, homographies_to_reference_frame)

  return  homographies_to_reference_frame, frames_to_transform





def show_panorama_from_first_frame(folder, initial_to_frame, initial_from_frame, frame_increment, total_number_of_frames):
  
  homographies_to_reference_frame, frames_to_transform = homographies_to_first_frame(folder, initial_to_frame, initial_from_frame, frame_increment, total_number_of_frames)
  
  ref_im = cv2.imread(frames_to_transform[0])  # index 0 if reference frame is the first frame
  
  h = 1
  for H in homographies_to_reference_frame[1:]:
    
    from_im = cv2.imread(frames_to_transform[h])
    panorama = place_image2(ref_im, from_im, H) 
    cv2.imshow('panorama', panorama)
    
    ref_im = panorama
    h += 1
  
  return panorama
  
  
  

def show_panorama_iso_from_first_frame(folder, initial_to_frame, initial_from_frame, frame_increment, total_number_of_frames, frame_index_to_iso):
  
  homographies_to_reference_frame, frames_to_transform = homographies_to_first_frame(folder, initial_to_frame, initial_from_frame, frame_increment, total_number_of_frames)
  
  ref_im = cv2.imread(frames_to_transform[0])  # index 0 if reference frame is the first frame
  
  h = 1
  for H in homographies_to_reference_frame[1:]:
    
    from_im = cv2.imread(frames_to_transform[h])
    panorama = place_image2(ref_im, from_im, H) 
    
    ref_im = panorama
    h += 1
 
 
  frame_to_iso = cv2.imread(frames_to_transform[frame_index_to_iso])
  (ht, width) = panorama.shape[:2]
  iso_image = cv2.warpPerspective(frame_to_iso, homographies_to_reference_frame[frame_index_to_iso], (width, ht))
    
  cv2.imshow('iso_image', iso_image)
  
  return iso_image
  
  
  ###################################
      
      
def getHomographiesToReferenceFrame_At_Center(interFrame_homographies, homogs_to_reference_frame, center_index):
  
      
  for i in range(center_index):
    h_to_reference = interFrame_homographies[i]
    j = i
    while(j < center_index - 1):
      h_to_reference = np.dot(interFrame_homographies[j+1], h_to_reference)
      j += 1 
    homogs_to_reference_frame.append(h_to_reference)
    
 
  homogs_to_reference_frame.append(np.identity(3))  # center frame's homography to reference frame (identity matrix)
  
  for h in interFrame_homographies[center_index: ]:
    homogs_to_reference_frame.append(np.dot(homogs_to_reference_frame[-1], h))
    
  
  return homogs_to_reference_frame
  
  
  
def homographies_to_center_frame(folder, initial_to_frame, initial_from_frame, center_frame, frame_increment, total_number_of_frames):
  
  
  from_frm = 'frame' + str(initial_from_frame) + '.jpg'
  from_frm = folder + from_frm
  
  to_frm = 'frame' + str(initial_to_frame) + '.jpg'
  to_frm = folder + to_frm
  
  
  to_frame = cv2.imread(to_frm)
  from_frame = cv2.imread(from_frm)
  interFrame_homographies = []
  interFrame_homographies.append(getHomography(to_frame, from_frame))
  
  
  homographies_to_reference_frame = []
  frames_to_transform = [from_frm, to_frm]
  
    
  i = initial_to_frame + frame_increment  # increment to next frame in series
  
  while(i <= center_frame):     
    
    from_frame = to_frame[:] # places a copy of to_frame into from_frame
    next_jpg = 'frame' + str(i) + '.jpg'
    next_jpg = folder + next_jpg
    to_frame = cv2.imread(next_jpg)
    interFrame_homographies.append(getHomography(to_frame, from_frame))
    
    frames_to_transform.append(next_jpg)
    
    i += frame_increment  # increment size
    
  
    
  while(i <= total_number_of_frames): 

    next_jpg = 'frame' + str(i) + '.jpg'
    next_jpg = folder + next_jpg
    from_frame = cv2.imread(next_jpg)
    interFrame_homographies.append(getHomography(to_frame, from_frame))
     
    frames_to_transform.append(next_jpg)
    to_frame = from_frame[:] # places a copy of from_frame into to_frame
    
    i += frame_increment  # increment size
    
  if initial_from_frame < 50:
    center_frame_index = (total_number_of_frames/2)/frame_increment
  else:
    n =   (total_number_of_frames - initial_from_frame)/2
    center_frame_index = int(n/frame_increment)
  
  homographies_to_reference_frame = getHomographiesToReferenceFrame_At_Center(interFrame_homographies, homographies_to_reference_frame, center_frame_index)

  return  homographies_to_reference_frame, frames_to_transform
  


def show_panorama_from_center_frame(folder, initial_to_frame, initial_from_frame, center_frame, frame_increment, total_number_of_frames):
  
  homographies_to_reference_frame, frames_to_transform = homographies_to_center_frame(folder, initial_to_frame, initial_from_frame, center_frame, frame_increment, total_number_of_frames)

  
  far_left_img = cv2.imread(frames_to_transform[0])  # index 0 if reference frame is the first frame
  
  far_left_homography = homographies_to_reference_frame[0]
  offset = -(far_left_homography[0,2]/far_left_homography[2,2])  # the offset of the far left frame point(0,0) from the center frame's point (0,0)
  
  far_left_homography[0,2] += offset
  
  size = calculate_size(far_left_img.shape[:2], far_left_homography) 
  panorama = cv2.warpPerspective(far_left_img, far_left_homography, size)
  
  cv2.imshow('panorama', panorama)
  i = 1

  for H in homographies_to_reference_frame[1:]:
    
    next_img = cv2.imread(frames_to_transform[i])
    panorama = place_image3(panorama, next_img, H, offset) 
    cv2.imshow('panorama', panorama)
    
    i += 1
    
  return panorama


  
  
def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)
  
  
  
  
  
  
  
  
if __name__ == "__main__":
   
  print "Go Moose!"
 
  # Select range of frames to construct background mosaic
  #  FROM FIRST FRAME   
  folder = 'frames_folder/'
  initial_to_frame = 125
  initial_from_frame = 130
  frame_increment = 5
  highest_frame_number = 180
  

  
  
  ### Output background mosaic to folder
  background_mosaic = show_panorama_from_first_frame(folder, initial_to_frame, initial_from_frame, frame_increment, highest_frame_number)
  cv2.imwrite("folder for frames/background_mosaic.jpg", background_mosaic)

'''
  ### Output warped frames to folder
  index = 17
  frame_index_to_isolate = index
  isolated_transformed_frame = show_panorama_iso_from_first_frame(folder, initial_to_frame, initial_from_frame, frame_increment, highest_frame_number, frame_index_to_isolate)
  cv2.imwrite("folder_for_warped_frames/frame_" + str(index) + ".jpg", isolated_transformed_frame)
  





### Segment warped frames using background mosaic
 #t1 = cv2.imread('frames_folder/warped_image.jpg')
 #t2 = cv2.imread('frames_folder/background_image.jpg')

###  OR

### Segment frame i using frame i-1 and frame i+1
  img1 = cv2.cvtColor(cv2.imread('frames_folder/frame_1.jpg'), cv2.COLOR_RGB2GRAY)
  img2 = cv2.cvtColor(cv2.imread('frames_folder/frame_2.jpg'), cv2.COLOR_RGB2GRAY)
  img3 = cv2.cvtColor(cv2.imread('frames_folder/frame_3.jpg'), cv2.COLOR_RGB2GRAY)
  segmented = diffImg(img1, img2, img3)
  
  ### clean up segnented binary image
  ret,thresh1 = cv2.threshold(segmented,20,255,cv2.THRESH_BINARY)
  kernel = np.ones((5,5),np.uint8)
  #erosion = cv2.erode(thresh1,kernel,iterations = 1)
  #dilation = cv2.dilate(thresh1,kernel,iterations = 1)
  #opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel) # erosion followed by dilation
  closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel) # dilation followed by erosion
  
  ### write segmented silhouette to folder
  #cv2.imwrite("Frames to jpg (Central)/pano_male.jpg", closing)

  
  #show('diff', thresh1)
  show('diff', closing)
  
  
  
  
  import sys, select 
  print "Press enter or any key on one of the images to exit"
  while True:
    if cv2.waitKey(100) != -1:
      break
    
    i, o, e = select.select( [sys.stdin], [], [], 0.1 )
    if i:
      break
 '''