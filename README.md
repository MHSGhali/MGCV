# MGCV [WIP]
This repository contains a Python-based image processing pipeline for performing various operations on images. The pipeline is organized into several modules for different tasks. 

Folder Structure

.
├── README.md
├── run.py
├── stacked_image.jpg
├── LICENSE.txt  
└── two_dimensions
    ├── filters.py
    ├── helper.py
    ├── processing.py
    └── __init__.py


## Getting Started

To run the image processing pipeline, follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have the required Python packages installed. You can install them using the following command:
   ```
   pip install -r requirements.txt

3. Open the run.py script and modify the image_path and file_type variables to point to your image dataset and specify the file type you're working with.
4. Run the run.py script:
    ```
    python run.py

5. The processed and stacked image will be saved as stacked_image.jpg.

## Modules

### filters.py

This module contains the `Filters` class, which handles various filtering operations on images. The class initializes with the image path and file type and provides methods for performing edge detection, masking, and stacking operations.

#### Infinity Focus (Focus Stacking)
The goal of this pipeline is to stack images with random focus areas intoi a single hyper-focused image.

### helper.py

The `Helpers` class in this module offers utility functions for loading images, plotting Gaussian pyramids, and displaying images.

### processing.py

The `ImageProcessing` class in this module provides methods for performing image filtering, thresholding, and other image processing techniques.

### Usage and Results

The `run.py` script demonstrates the usage of the pipeline. It loads images from the specified `image_path`, applies edge detection using the Sobel filter, performs masking and stacking, and saves the stacked image as `stacked_image.jpg`.

### Examples

Using the Light field Salency Dataset (LFSD) 
https://www.eecis.udel.edu/~nianyi/LFSD.htm

### Contributing

Contributions to this repository are welcome! Feel free to open issues or pull requests for any improvements or additional features.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

