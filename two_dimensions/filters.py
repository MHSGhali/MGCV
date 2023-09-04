import os
import time

from .processing import ImageProcessing
from .homography import Homography
from .helper import Helpers
import numpy as np
import skimage as sk

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
        
    def panorama(self):
        image1, image2 = sk.img_as_ubyte(self.images[0]), sk.img_as_ubyte(self.images[1])
        scale = 0.5
        im1, im2 =  self.help.scale_image(image1, scale),  self.help.scale_image(image2, scale)
        locs, decs = self.homography.briefLite(im1, im2, visual = True)
        matches = self.homography.briefMatch(decs[0], decs[1])
        np.random.seed(0)
        H2to1 = self.homography.ransac_homography(matches, locs[0], locs[1], num_iter= 1000, threshold=1)
        pano_im = self.homography.imageStitching(im1, im2, H2to1)
        return pano_im.astype(np.float32)