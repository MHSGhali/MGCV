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

    def fill_between_first_last_true_window(self, binary, window_size, stride):
        num_rows = (binary.shape[0] - window_size) // stride + 1
        num_cols = (binary.shape[1] - window_size) // stride + 1
        filled_binary = np.copy(binary)
        
        for row in range(num_rows):
            for col in range(num_cols):
                window = binary[row * stride:row * stride + window_size,
                            col * stride:col * stride + window_size]
                
                filled_binary_row = np.copy(window)
                for r in range(window.shape[0]):
                    col_indices = np.where(window[r, :])[0]
                    if len(col_indices) > 0:
                        first_col = col_indices[0]
                        last_col = col_indices[-1]
                        filled_binary_row[r, first_col:last_col + 1] = True
                
                filled_binary_col = np.copy(window)
                for c in range(window.shape[1]):
                    row_indices = np.where(window[:, c])[0]
                    if len(row_indices) > 0:
                        first_row = row_indices[0]
                        last_row = row_indices[-1]
                        filled_binary_col[first_row:last_row + 1, c] = True
                
                filled_binary[row * stride:row * stride + window_size,
                            col * stride:col * stride + window_size] |= filled_binary_col | filled_binary_row
                
        return filled_binary
    
    def convex_hull_window(self, image, window_size, stride):
        num_rows = (image.shape[0] - window_size) // stride + 1
        num_cols = (image.shape[1] - window_size) // stride + 1
        combined_mask = np.zeros(image.shape, dtype=bool)
        for row in range(num_rows):
            for col in range(num_cols):
                window = image[row * stride:row * stride + window_size,
                            col * stride:col * stride + window_size]
                non_zero_indices = np.transpose(np.nonzero(window))
                if len(non_zero_indices) > 0:
                    hull_mask = sk.morphology.convex_hull_image(window)
                    combined_mask[row * stride:row * stride + window_size,
                                col * stride:col * stride + window_size] |= hull_mask
        return combined_mask
    
    def fill_voids(self, binary_mask):
        filled_mask = sk.morphology.remove_small_holes(binary_mask, area_threshold=10)
        return filled_mask