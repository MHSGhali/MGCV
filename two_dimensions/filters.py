import os
import time

from .processing import ImageProcessing
from .helper import Helpers
import numpy as np
import skimage as sk

class Filters():
    def __init__(self, image_path, file_type):
        start_time = time.time()
        self.help = Helpers(image_path, file_type)
        self.process = ImageProcessing()
        self.images = self.help.load_images()
        end_time = time.time()
        print(f'Initialization time = {int(round(1000 * (end_time - start_time)))} mili-seconds')

    def infinity_focus(self, method, window_num, stride_window):
        masks = []
        i = 1
        for img in self.images:
            start_time = time.time()
            img = sk.color.rgb2gray(img)

            if method == "sobel":
                filtered, _ = self.process.sobel(img)
            elif method == "gaussian":
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
        masks, not_masked = self.process.clean_masks(self.images, masks, method = 'group')
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
        
        