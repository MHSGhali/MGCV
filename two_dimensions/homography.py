import os
import time

from .processing import ImageProcessing
from .helper import Helpers
import numpy as np
import math
import skimage as sk
from scipy.spatial.distance import cdist
import cv2
import matplotlib.pyplot as plt
from itertools import combinations

class Homography():
    def __init__(self):
        pass

    def briefLite(self, img1, img2, visual=False):
        FAST = cv2.FastFeatureDetector_create() 
        BRIEF = cv2.ORB_create()
        demo_ip = FAST.detect(img1, None)
        test_demo_ip = FAST.detect(img2, None)
        demo_ip, demo_descriptor = BRIEF.compute(img1, demo_ip)
        test_demo_ip, test_demo_descriptor = BRIEF.compute(img2, test_demo_ip)
        
        if visual: 
            visual_img = img1.copy()
            test_visual_img = img2.copy()
            cv2.drawKeypoints(visual_img, demo_ip, visual_img, color = (255, 0, 0))
            fx, plots = plt.subplots(1, 1, figsize=(20,10))
            _ = plots.set_title("Interest Points Detected in Original Image")
            _ = plots.imshow(visual_img, cmap='gray')

        return [demo_ip, test_demo_ip], [demo_descriptor, test_demo_descriptor]
    
    def briefMatch(self, desc1, desc2, ratio=0.8):     
        D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
        # find shortested distance
        ix2 = np.argmin(D, axis=1)
        d1 = D.min(1)
        
        # find second smallest distance
        d12 = np.partition(D, 2, axis=1)[:,0:2]
        d2 = d12.max(1)
        r = d1 / (d2 + 1e-10)
        is_discr = r < ratio
        ix2 = ix2[is_discr]
        ix1 = np.arange(D.shape[0])[is_discr]
        matches = np.stack((ix1,ix2), axis=-1)
        return matches
    
    def compute_homography(self, p1, p2):
        assert p1.shape[1] == p2.shape[1]
        assert p1.shape[0] == 2
        
        #############################
        # A_i:
        #  -x  -y  -1   0   0   0  xx'  yx'  x'
        #   0   0   0  -x  -y  -1  xy'  yy'  y'
        #############################
        
        A = np.zeros((2*p1.shape[1], 9)) #2N*9
        num_points = p1.shape[1]
    
        A[:num_points, 0] = -p1[0, :]
        A[:num_points, 1] = -p1[1, :]
        A[:num_points, 2] = -1
        A[:num_points, 6] = p1[0, :] * p2[0, :]
        A[:num_points, 7] = p1[1, :] * p2[0, :]
        A[:num_points, 8] = p2[0, :]

        A[num_points:, 3] = -p1[0, :]
        A[num_points:, 4] = -p1[1, :]
        A[num_points:, 5] = -1
        A[num_points:, 6] = p1[0, :] * p2[1, :]
        A[num_points:, 7] = p1[1, :] * p2[1, :]
        A[num_points:, 8] = p2[1, :]
        
        _, _, vh = np.linalg.svd(A) 
        H2to1 = vh[-1,:].reshape((3, 3)) 
        H2to1 = H2to1 / H2to1[2][2] 
            
        return H2to1
    
    def ransac_homography(self,matches, locs1, locs2, num_iter=5000, threshold=2):
        #################################################################
        # convert between OpenCV keypoint and match object to NumPy array
        #################################################################
        pts1, pts2 = [], []
        for match in matches:
            p1 = locs1[match[0]].pt
            p2 = locs2[match[1]].pt
            pts1.append(p1)
            pts2.append(p2)
        locs1 = np.array(pts1)
        locs2 = np.array(pts2)
        x1, x2 = np.vstack((locs1.T, np.ones(locs1.shape[0]))), np.vstack((locs2.T, np.ones(locs2.shape[0])))
        
        bestH = np.zeros((3,3)) # to be implemented, should be an empty 3x3 NumPy array
        # RANSAC parameters
        max_inliers = -1
        max_selected_points = None
        # start RANSAC loop
        for _ in range(num_iter):
            picked_idx = np.random.choice(locs1.shape[0], 4)
            pts1, pts2 = locs1[picked_idx, ...].T, locs2[picked_idx, ...].T
            H = self.compute_homography(pts2, pts1) 
            # H = self.ndlt_homography(pts2, pts1) 
            x1_pred = H.dot(x2) 
            x1_pred /= x1_pred[2:3, ...] #normalize
            error = np.sqrt(np.sum((x1 - x1_pred) ** 2, axis=0)) 
            selected_points = np.where(error < threshold)[0]
            num_inliers = selected_points.size  
            if num_inliers > max_inliers:
                # update the best guess homography 
                max_inliers = num_inliers
                max_selected_points = selected_points
                bestH = H
        
        # finally, compute the homography again using the best inliers set
        pts1 = locs1[max_selected_points, ...].T
        pts2 = locs2[max_selected_points, ...].T
        bestH = self.compute_homography(pts2,pts1) 
        # bestH = self.ndlt_homography(pts2, pts1) 
        return bestH
    
    def imageStitching(self, im1, im2, H2to1):
        scale = 1
        tx = 0
        M_scale = np.array([[scale, 0, tx], [0, scale, 0], [0, 0, 1]], dtype=np.float32) # to be implemented, recall how the 3x3 scaling matrix should looks like in homogeneous coordinate

        # vectorize corner calcuation, corner should be the most extreme point from the image, which is (0, 0) and (imgh, imgw)
        im2_h, im2_w = im2.shape[0], im2.shape[1]
        corners = np.array([[0, 0], [0, im2_h], [im2_w, 0], [im2_w, im2_h]]).T
        wrapped_corner = (M_scale @ H2to1) @ np.concatenate([corners, np.ones((1, corners.shape[1]))])
        wrapped_corner /= np.reshape(wrapped_corner[2, ...], (1, wrapped_corner.shape[1]))
        wrapped_corner = wrapped_corner[0:2, ...]

        minh = np.min(wrapped_corner[1, ...])
        maxh = np.max(wrapped_corner[1, ...])
        minw = np.min(wrapped_corner[0, ...]) 
        maxw = np.max(wrapped_corner[0, ...])
        
        ty = max(0, -minh)
        tx = max(0, -minw)  # Updated this line
        M_translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32) # to be implemented, recall how the 3x3 translate matrix should looks like in homogeneous coordinate
        
        out_width = int(math.ceil(maxw - minw))  # Updated this line
        out_height = int(math.ceil(maxh - minh))
        out_size = (out_width, out_height)

        M = M_translate @ M_scale # to be implemented translate then scale
        pano_im1 = cv2.warpPerspective(im1, M, out_size).astype(np.float32) # to be implemented, utilize cv2.warpPerspective(im, H, out_size) function
        pano_im2 = cv2.warpPerspective(im2, M @ H2to1, out_size).astype(np.float32) # to be implemented, utilize cv2.warpPerspective(im, H, out_size) function
        pano_im1 /= 255.
        pano_im2 /= 255.
    
        im1_pano_mask =  pano_im1 > 0 #to be implemented, only left the part where pano_img1 has pixel value larger than 0
        im2_pano_mask = pano_im2 > 0 #to be implemented, only left the part where pano_img2 has pixel value larger than 0
        
        im_int_mask = im1_pano_mask & im2_pano_mask  # handle the center where 2 images meet
        pano_im_full = pano_im1 + pano_im2 #to be implemented, which is formed by pano_im1 and pano_im2

        # mask filter, where only left the park has value > 0 
        im_1 = pano_im_full * (~im1_pano_mask & im2_pano_mask)  
        im_2 = pano_im_full * (im1_pano_mask & ~im2_pano_mask) 
        
        ratio = 0.5   #to be implemented, blend the ceneter of the image, which ideally should apply a ratio
        im_int = pano_im_full * im_int_mask  * ratio
        pano_im = im_1 + im_int + im_2 #to be implemented, the final panoram should be combination of im_R, im_L and im_center
        
        return pano_im
    
    def normalize_points(self, points):
        if points.shape[1] == 0:
            return np.identity(3), points
        mean = np.mean(points, axis=1)
        scale = np.sqrt(2) / np.mean(np.linalg.norm(points - mean.reshape(-1, 1), axis=0))
        T = np.array([[scale, 0, -scale * mean[0]],
                    [0, scale, -scale * mean[1]],
                    [0, 0, 1]])

        points_homogeneous = np.vstack((points, np.ones(points.shape[1])))
        normalized_points_homogeneous = np.dot(T, points_homogeneous)
        normalized_points = normalized_points_homogeneous[:2, :]

        return T, normalized_points

    def ndlt_homography(self, points1, points2):
        if points1.shape[1] < 4 or points2.shape[1] < 4:
            raise ValueError("At least 4 corresponding points are required for NDLT.")
    
        T1, normalized_points1 = self.normalize_points(points1)
        T2, normalized_points2 = self.normalize_points(points2)

        A = []
        for i in range(normalized_points1.shape[1]):
            x1, y1 = normalized_points1[:, i]
            x2, y2 = normalized_points2[:, i]
            A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
            A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])

        A = np.array(A)

        _, _, vh = np.linalg.svd(A)
        H2to1 = vh[-1, :].reshape((3, 3))
        H2to1 = np.linalg.inv(T1) @ H2to1 @ T2  # Denormalize

        return H2to1

    def get_combination_homographies(self, images, scale, num_iter, threshold):
        homographies = []
        number_of_matches = [] 
        combination = []
        for images_set in combinations(range(len(images)), 2):
            combination.append(images_set)
            img1 = Helpers.scale_image(self, sk.img_as_ubyte(images[images_set[0]]), scale)
            img2 = Helpers.scale_image(self, sk.img_as_ubyte(images[images_set[1]]), scale)
            locs, decs = self.briefLite(img1, img2, visual=False)
            matches = self.briefMatch(decs[0], decs[1])
            number_of_matches.append(len(matches))
            np.random.seed(0)
            H2to1 = self.ransac_homography(matches, locs[0], locs[1], num_iter, threshold)
            homographies.append(H2to1)
        return homographies, number_of_matches, combination

    def plotMatches(self, im1, im2, matches, locs1, locs2):
        n_locs1, n_locs2 = [], []
        # convert locs to NumPy array
        for i in locs1:
            n_locs1.append([i.pt[0], i.pt[1]])
        for i in locs2:
            n_locs2.append([i.pt[0], i.pt[1]])
        locs1 = np.array(n_locs1)
        locs2 = np.array(n_locs2)
        
        fig = plt.figure()
        imH = max(im1.shape[0], im2.shape[0])
        im = np.zeros((imH, im1.shape[1]+im2.shape[1]))
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        im[0:im1.shape[0], 0:im1.shape[1]] = im1
        im[0:im2.shape[0], im1.shape[1]:] = im2
        plt.imshow(im, cmap='gray')
        for i in range(matches.shape[0]):
            pt1 = locs1[matches[i,0], 0:2]
            pt2 = locs2[matches[i,1], 0:2].copy()
            pt2[0] += im1.shape[1]
            x = np.array([pt1[0], pt2[0]])
            y = np.array([pt1[1], pt2[1]])
            plt.plot(x,y,'r',lw=1)
            plt.plot(x,y,'g.')
        plt.show()