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
    
    def average_mask_in_windows(self, mask, window_size):
        windows = sk.util.shape.view_as_windows(mask, (window_size, window_size))
        window_means = np.mean(windows, axis=(2, 3))  # Calculate mean within each window
        return sk.transform.resize(window_means, mask.shape, mode='constant', anti_aliasing=True)

    def fill_between_first_last_true(self, binary, window_size):
        filled_binary_row = np.copy(binary)
        for row in range(binary.shape[0]):
            col_indices = np.where(binary[row, :])[0]
            if len(col_indices) > 0:
                first_col = col_indices[0]
                last_col = col_indices[-1]
                filled_binary_row[row, first_col:last_col + 1] = True
        
        filled_binary_col = np.copy(binary)
        for col in range(binary.shape[1]):
            row_indices = np.where(binary[:, col])[0]
            if len(row_indices) > 0:
                first_row = row_indices[0]
                last_row = row_indices[-1]
                filled_binary_col[first_row:last_row + 1, col] = True
        
        return self.average_mask_in_windows(filled_binary_col * filled_binary_row, window_size)