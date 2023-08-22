import os

from .processing import ImageProcessing
from .helper import Helpers
import numpy as np
import skimage as sk

class Filters():
    def __init__(self, image_path, file_type, sigma=10, truncate = 4):
        self.help = Helpers(image_path, file_type)
        self.process = ImageProcessing()
        self.images = self.help.load_images()
        self.sigma, self.truncate = sigma, truncate

    def infinity_focus(self):
        filters1 = []
        filters2 = []
        for img in self.images:
            img = sk.color.rgb2gray(img)
            
            _, residual = self.process.gaussian(img, self.sigma, self.truncate)
            normalized = self.process.normalize_image(residual,0,255)
            thresh = self.process.get_threshold(normalized)
            binary1 =  normalized >= thresh
            
            filtered, _ = self.process.sobel(img)
            normalized = self.process.normalize_image(filtered,0,255)
            thresh = self.process.get_threshold(normalized)
            binary2 =  normalized >= thresh

            filter1 = self.process.fill_between_first_last_true(binary1, 25)
            filter2 = self.process.fill_between_first_last_true(binary2, 25)

            filters1.append(filter1 * img)
            filters2.append(filter2 * img)
            
        self.help.show_images(filters1, filters2)