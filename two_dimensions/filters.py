import os
import time

from .processing import ImageProcessing
from .homography import Homography
from .helper import Helpers
from itertools import groupby
import numpy as np
import skimage as sk
import cv2
import matplotlib.pyplot as plt


class Filters():
    def __init__(self, image_path, file_type):
        start_time = time.time()
        self.help = Helpers(image_path, file_type)
        self.process = ImageProcessing()
        self.homography = Homography()
        self.images = self.help.load_images()
        end_time = time.time()
        print(f'Initialization time = {int(round(1000 * (end_time - start_time)))} mili-seconds')

    def compensate_focal_distance(self):
        for i,img in enumerate(self.images):
            img = sk.img_as_ubyte(sk.color.rgb2gray(img))
            if i == 0:
                img1 = img
            else:
                homography = self.homography.macthed_image(img,img1)
                self.images[i] = cv2.warpPerspective(self.images[i], homography, (img1.shape[1], img1.shape[0]),flags=cv2.INTER_LINEAR)
                img1 = cv2.warpPerspective(img, homography, (img1.shape[1], img1.shape[0]),flags=cv2.INTER_LINEAR)
    
    def infinity_focus(self, edge_method, window_num, stride_window, cluster_method = 'group'):
        start_time = time.time()
        self.compensate_focal_distance()
        end_time = time.time()
        print(f'Homography Matching time = {int(round(1000 * (end_time - start_time)))} mili-seconds')
        
        masks = []
        for i,img in enumerate(self.images):
            start_time = time.time()
            img = sk.color.rgb2gray(img)
            
            if edge_method == "sobel":
                filtered, _ = self.process.sobel(img)
            elif edge_method == "gaussian":
                _, filtered = self.process.gaussian(img)
            
            normalized = self.process.normalize_image(filtered)
            binary = self.process.get_threshold(normalized)

            hull = self.process.convex_hull_window(binary, window_num, stride_window)
            filled_hull = self.process.fill_voids(hull)
            masks.append(filled_hull)
            end_time = time.time()
            print(f'Edge Detection and Masking time of image {i} = {int(round(1000 * (end_time - start_time)))} mili-seconds')
        
        start_time = time.time()
        masks = self.process.clean_masks(self.images, masks, method = cluster_method)
        end_time = time.time()
        print(f'Mask Cleaning time = {int(round(1000 * (end_time - start_time)))} mili-seconds')

        start_time = time.time()
        result_images = self.process.masked_images(self.images, masks)
        end_time = time.time()
        print(f'Masking Images time = {int(round(1000 * (end_time - start_time)))} mili-seconds')
        
        start_time = time.time()
        stacked_image = self.process.stacked_image(self.images, result_images, masks)
        end_time = time.time()
        print(f'Image Stacking time = {int(round(1000 * (end_time - start_time)))} mili-seconds')
        return stacked_image, masks
    
    def opencv_stitch(self):
        start_time = time.time()
        stitching = cv2.Stitcher.create()
        images = []
        for image in self.images:
            images.append(sk.img_as_ubyte(image)[:, :, ::-1])
        _, pano_im = stitching.stitch(images)
        result = self.homography.remove_void_regions(pano_im)
        end_time = time.time()
        print(f'Panorama Stitching time = {int(round(1000 * (end_time - start_time)))} mili-seconds')
        return result