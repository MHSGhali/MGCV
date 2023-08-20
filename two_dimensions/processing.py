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
        return filtered_image
    
    def laplacian(self, image, k_size):
        filtered_image =  sk.filters.laplace(image, ksize=k_size)
        return filtered_image

    def hysteresis(self, edges, low, high):
        filtered_image = sk.filters.apply_hysteresis_threshold(edges, low, high)
        return filtered_image
    
    def laplacianPyramid(self, image, sigma_, truncate_, smallest_dim):
        image = image.astype(float) / 255
        image_pyramid = []
        gaussian_pyramid = []
        image_pyramid.append(image)
        while max(np.shape(image)) > smallest_dim:
            temp = self.gaussian(image, sigma_, truncate_)
            residual = image - temp 
            gaussian_pyramid.append(residual)
            image = image[::2, ::2]
            image_pyramid.append(image)
        return image_pyramid, gaussian_pyramid