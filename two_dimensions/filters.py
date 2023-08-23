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

            filtered, _ = self.process.sobel(img)
            normalized = self.process.normalize_image(filtered,0,255)
            thresh = self.process.get_threshold(normalized)
            binary =  normalized >= thresh
            hull = self.process.convex_hull_window(binary, 50, 5)
            fill = self.process.fill_between_first_last_true_window(binary, 50, 5)
            filters2.append(self.process.fill_voids(fill) * img)
            filters1.append(self.process.fill_voids(hull) * img)
            
        self.help.show_images(filters1, filters2)