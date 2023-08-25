import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

class ImageProcessing():
    def __init__(self):      
        pass
    
    def gaussian(self, image, sigma_ = 10, truncate_ = 4):
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
        binary =  image >= threshold
        return binary

    def normalize_image(self, image):
        min_value, max_value = 0, 255
        min_pixel, max_pixel = np.min(image), np.max(image)
        normalized_image = (image - min_pixel) / (max_pixel - min_pixel)
        normalized_image = normalized_image * (max_value - min_value) + min_value
        return normalized_image.astype(np.uint8)

    def laplacianPyramid(self, image, sigma_ = 10, truncate_ = 4, smallest_dim = 360):
        image = image.astype(float) / 255
        gaussian_pyramid = []
        while max(np.shape(image)) > smallest_dim:
            _, res = self.gaussian(image, sigma_, truncate_)
            gaussian_pyramid.append(res)
            image = image[::2, ::2]
        return image, gaussian_pyramid

    def combine_masks(self, results, masks):
        canvas_height = max(image.shape[0] for image in results)
        canvas_width = max(image.shape[1] for image in results)
        combined_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        for masked_image, binary_mask in zip(results, masks):
            row_indices, col_indices = np.where(binary_mask)
            combined_image[row_indices, col_indices, :] = masked_image[row_indices, col_indices, :]
        return combined_image
    
    def convex_hull_window(self, image, window_div, stride_div):
        window_size = min([x//window_div for x in np.shape(image)])
        stride = window_size//stride_div
        num_rows = (image.shape[0] - window_size) // stride + 1
        num_cols = (image.shape[1] - window_size) // stride + 1
        combined_mask = np.zeros(image.shape, dtype=bool)
        for row in range(num_rows):
            for col in range(num_cols):
                window = image[row * stride:row * stride + window_size, col * stride:col * stride + window_size]
                non_zero_indices = np.transpose(np.nonzero(window))
                if len(non_zero_indices) > 0:
                    hull_mask = sk.morphology.convex_hull_image(window)
                    combined_mask[row * stride:row * stride + window_size, col * stride:col * stride + window_size] |= hull_mask
        return combined_mask
    
    def fill_voids(self, binary_mask):
        seed = np.copy(binary_mask)
        seed[1:-1, 1:-1] = binary_mask.max()
        filled_mask = sk.morphology.reconstruction(seed, binary_mask, method='erosion')
        return filled_mask