import os

from .processing import ImageProcessing
from .helper import Helpers
import numpy as np
import skimage as sk

class Filters():
    def __init__(self, image_path, file_type):
        self.help = Helpers(image_path, file_type)
        self.process = ImageProcessing()
        self.images = self.help.load_images()

    def infinity_focus(self, method, window_num, stride_window):
        masks = []
        for i, img in enumerate(self.images):
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
        
        masks = self.process.clean_masks(self.images, masks)
        result_images = self.process.masked_images(self.images, masks)
        stacked_image = self.process.stacked_image(result_images, masks)
        return stacked_image, masks
        
        