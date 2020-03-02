import cv2
import numpy as np
import imutils
#import tqdm
import os
import sys
#from moviepy.editor import ImageSequenceClip


class VideoStitcher:
    def __init__(self):
        # Initialize argume

        # Initialize the saved homography matrix
        self.isv3 = imutils.is_cv3()
        self.smoothing_window_size=400
        
        self.saved_homo_matrix = None
        
        
    def create_mask(self,img1,img2,version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])     
        
    def blending(self,img1,img2,H):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1,img2,version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1,img2,version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
        result=panorama1+panorama2

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result    

    def stitch(self, images, ratio=0.75, reproj_thresh=30.0):
        # Unpack the images
        (image_b, image_a) = images

        # If the saved homography matrix is None, then we need to apply keypoint matching to construct it
        if self.saved_homo_matrix is None:
            # Detect keypoints and extract
            (keypoints_a, features_a) = self.detect_and_extract(image_a)
            (keypoints_b, features_b) = self.detect_and_extract(image_b)

            # Match features between the two images
            matched_keypoints = self.match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)

            # If the match is None, then there aren't enough matched keypoints to create a panorama
            #cv2.imshow("correspondences", matched_keypoints[0])
            if matched_keypoints is None:
                return None

            # Save the homography matrix
            self.saved_homo_matrix = matched_keypoints[1]
           
           
        # Apply a perspective transform to stitch the images together using the saved homography matrix
        #output_shape = (image_a.shape[1] + image_b.shape[1], image_a.shape[0])
        #result = cv2.warpPerspective(image_a, self.saved_homo_matrix, output_shape)
        #result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b
        height_img1 = image_a.shape[0]
        width_img1 = image_a.shape[1]
        width_img2 = image_b.shape[1]
        height = height_img1
        width  = width_img1 +width_img2  
        #width = image_a.shape[1] + image_b.shape[1]
        #height = image_a.shape[0] + image_b.shape[0]

        result = cv2.warpPerspective(image_a, self.saved_homo_matrix, (width, height))
        result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b
        #foreground = image_a
        #background = image_b
        #alpha = result
        #while foreground.isOpened():
        #fr_foreground = foreground/255.0
       # fr_background = background/255.0     
        #fr_alpha = alpha/255.0

        #result1 = fr_foreground*fr_alpha+fr_background*(1-fr_alpha)
         #result = self.blending(image_a,image_b,self.saved_homo_matrix) 
        '''
        image_a = image_a.astype(float)
        image_b = image_b.astype(float)
        alpha = result
  
# Normalize the alpha mask to keep intensity between 0 and 1
        alpha = alpha.astype(float)/255
 
# Multiply the foreground with the alpha matte
        image_a = cv2.multiply(alpha, image_a)
 
# Multiply the background with ( 1 - alpha )
        image_b = cv2.multiply(1.0 - alpha, image_b)
        outImage = cv2.add(image_a, image_b)
        '''
        # Return the stitched image
        return result
    
 
        
    @staticmethod
    def detect_and_extract(image):
        # Detect and extract features from the image (DoG keypoint detector and SIFT feature extractor)
        descriptor = cv2.xfeatures2d.SIFT_create()
        (keypoints, features) = descriptor.detectAndCompute(image, None)

        # Convert the keypoints from KeyPoint objects to numpy arrays
        keypoints = np.float32([keypoint.pt for keypoint in keypoints])

        # Return a tuple of keypoints and features
        return (keypoints, features)

    @staticmethod
    def match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh):
        # Compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(features_a, features_b, k=2)
        matches = []

        for raw_match in raw_matches:
            # Ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if len(raw_match) == 2 and raw_match[0].distance < raw_match[1].distance * ratio:
                matches.append((raw_match[0].trainIdx, raw_match[0].queryIdx))

        # Computing a homography requires at least 4 matches
        if len(matches) > 4:
            # Construct the two sets of points
            points_a = np.float32([keypoints_a[i] for (_, i) in matches])
            points_b = np.float32([keypoints_b[i] for (i, _) in matches])

            # Compute the homography between the two sets of points
            (homography_matrix, status) = cv2.findHomography(points_a, points_b, cv2.RANSAC, reproj_thresh)

		  
            # Return the matches, homography matrix and status of each matched point
            return (matches, homography_matrix, status)

        # No homography could be computed
        return None

    @staticmethod
    def draw_matches(image_a, image_b, keypoints_a, keypoints_b, matches, status):
        # Initialize the output visualization image
        (height_a, width_a) = image_a.shape[:2]
        (height_b, width_b) = image_b.shape[:2]
        visualisation = np.zeros((max(height_a, height_b), width_a + width_b, 3), dtype="uint8")
        visualisation[0:height_a, 0:width_a] = image_a
        visualisation[0:height_b, width_a:] = image_b

        for ((train_index, query_index), s) in zip(matches, status):
            # Only process the match if the keypoint was successfully matched
            if s == 1:
                # Draw the match
                point_a = (int(keypoints_a[query_index][0]), int(keypoints_a[query_index][1]))
                point_b = (int(keypoints_b[train_index][0]) + width_a, int(keypoints_b[train_index][1]))
                cv2.line(visualisation, point_a, point_b, (0, 255, 0), 1)

        # return the visualization
        return visualisation
