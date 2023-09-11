import os
import time

from .processing import ImageProcessing
from .homography import Homography
from .helper import Helpers
import numpy as np
import skimage as sk
import cv2


class Filters():
    def __init__(self, image_path, file_type):
        start_time = time.time()
        self.help = Helpers(image_path, file_type)
        self.process = ImageProcessing()
        self.homography = Homography()
        self.images = self.help.load_images()
        end_time = time.time()
        print(f'Initialization time = {int(round(1000 * (end_time - start_time)))} mili-seconds')

    def infinity_focus(self, edge_method, window_num, stride_window, cluster_method = 'group'):
        masks = []
        i = 1
        for img in self.images:
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
            i += 1
        
        start_time = time.time()
        masks, not_masked = self.process.clean_masks(self.images, masks, method = cluster_method)
        end_time = time.time()
        print(f'Mask Cleaning time = {int(round(1000 * (end_time - start_time)))} mili-seconds')

        start_time = time.time()
        result_images = self.process.masked_images(self.images, masks)
        end_time = time.time()
        print(f'Masking Images time = {int(round(1000 * (end_time - start_time)))} mili-seconds')

        start_time = time.time()
        stacked_image = self.process.stacked_image(self.images, result_images, masks, not_masked)
        end_time = time.time()
        print(f'Image Stacking time = {int(round(1000 * (end_time - start_time)))} mili-seconds')
        return stacked_image, masks
        
    def panorama(self, scale=1.0, num_iter=50000, threshold=10):
        homographies, number_of_matches, combination = self.homography.get_combination_homographies(self.images, scale, num_iter, threshold)
        if len(number_of_matches) < 3:
            homography_indices = [ind for ind, _ in enumerate(number_of_matches)]
        else:
            homography_indices = [ind for ind, x in enumerate(number_of_matches) if x > np.mean(number_of_matches)]
        
        matched_images = [combination[i] for i in homography_indices]
        matched_homographies = [homographies[i] for i in homography_indices]
        panorama_images = []

        for idx, set in enumerate(matched_images):
            img1 = self.help.scale_image(sk.img_as_ubyte(self.images[set[0]]), scale)
            img2 = self.help.scale_image(sk.img_as_ubyte(self.images[set[1]]), scale)
            H2to1 = matched_homographies[idx]
            panorama = sk.img_as_ubyte(self.homography.imageStitching(img1, img2, H2to1))
            panorama_images.append(panorama) 
    
        return panorama_images
    
    def opencv_stitch(self):
        stitching = cv2.Stitcher.create()
        images = []
        for image in self.images:
            images.append(sk.img_as_ubyte(image)[:, :, ::-1])
        _, pano_im = stitching.stitch(images)
        
        result = self.homography.remove_void_regions(pano_im)
        return result