import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

class ImageProcessing():
    def __init__(self):      
        pass
    
    def gaussian(self, image, sigma_, truncate_):
        filtered_image = sk.filters.gaussian(image, sigma=sigma_, truncate=truncate_, channel_axis = -1)
        residual_image = image - filtered_image
        return filtered_image, residual_image
    
    def laplacian(self, image, k_size):
        filtered_image =  sk.filters.laplace(image, ksize=k_size)
        residual_image = image - filtered_image
        return filtered_image, residual_image

    def hysteresis(self, edges, low, high):
        filtered_image = sk.filters.apply_hysteresis_threshold(edges, low, high)
        return filtered_image
    
    def sobel(self, image):
        filtered_image = sk.filters.sobel(image)
        residual_image = image - filtered_image
        return filtered_image, residual_image
    
    def get_threshold(self, image):
        threshold = sk.filters.threshold_otsu(image)
        return threshold
    
    def normalize_image(self, image, min_value=0, max_value=255):
        min_pixel, max_pixel = np.min(image), np.max(image)
        normalized_image = (image - min_pixel) / (max_pixel - min_pixel)
        normalized_image = normalized_image * (max_value - min_value) + min_value
        return normalized_image.astype(np.uint8)

    def laplacianPyramid(self, image, sigma_, truncate_, smallest_dim):
        image = image.astype(float) / 255
        image_pyramid = []
        gaussian_pyramid = []
        image_pyramid.append(image)
        while max(np.shape(image)) > smallest_dim:
            filt, res = self.gaussian(image, sigma_, truncate_)
            gaussian_pyramid.append(res)
            image = image[::2, ::2]
            image_pyramid.append(image)
        return image_pyramid, gaussian_pyramid