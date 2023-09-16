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
import imutils

class Homography():
    def __init__(self):
        pass

    def remove_void_regions(self, stitched_img):
        stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
        gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)
        mask = np.zeros(thresh_img.shape, dtype="uint8")
        x, y, w, h = cv2.boundingRect(areaOI)
        
        cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)
        minRectangle = mask.copy()
        sub = mask.copy()
        
        while cv2.countNonZero(sub) > 0:
            minRectangle = cv2.erode(minRectangle, None)
            sub = cv2.subtract(minRectangle, thresh_img)
        
        contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(areaOI)
        stitched_img = stitched_img[y:y + h, x:x + w]
        return stitched_img

    def macthed_image(self,img1,img2):
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        ratio = 0.75  # Adjust this threshold as needed
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return homography_matrix