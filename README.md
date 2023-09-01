# MGCV [WIP]
This repository contains a Python-based image processing pipeline for performing various operations on images. The pipeline is organized into several modules for different tasks. 

Folder Structure
```
.
│   LISCENSE.txt
│   README.md
│   requirements.txt
│   run.py
│   stacked_image.jpg
│
├───Test
│   ├───Images
│   │       1__refocus_0N.jpg
│   │       
│   ├───Masks
│   │       grouping.png
│   │       segmentation.png
│   │
│   └───Results
│           grouping.jpg
│           segmentation.jpg
│
└───two_dimensions
    │   filters.py
    │   helper.py
    │   processing.py
    │   __init__.py
```

## Getting Started

To run the image processing pipeline, follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have the required Python packages installed. You can install them using the following command:
   ```
   pip install -r requirements.txt
   ```

3. Open the run.py script and modify the image_path and file_type variables to point to your image dataset and specify the file type you're working with.
4. Run the run.py script:
    ```
    python run.py
    ```
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

<div style="display: flex; justify-content: center;">
  <table style="border-collapse: collapse;">
    <tr>
      <th colspan="3" style="text-align: center;"> Random Focus Images </th>
      <th colspan="1" style="text-align: center;"> Grouping Algorithm </th>
      <th colspan="1" style="text-align: center;"> Segmentation Algorithm </th>
    </tr>
    <tr>
        <td><img src=".\Test\Images\1__refocus_00.jpg" alt="Image 1" style="width: 100px;"></td>
        <td><img src=".\Test\Images\1__refocus_01.jpg" alt="Image 2" style="width: 100px;"></td>
        <td><img src=".\Test\Images\1__refocus_02.jpg" alt="Image 3" style="width: 100px;"></td>
        <td rowspan="3"><img src=".\Test\Results\grouping.jpg" alt="Resultant Image 1" style="width: 330px;"></td>
        <td rowspan="3"><img src=".\Test\Results\segmentation.jpg" alt="Resultant Image 2" style="width: 330px;"></td>
    </tr>
    <tr>
        <td><img src=".\Test\Images\1__refocus_03.jpg" alt="Image 4" style="width: 100px;"></td>
        <td><img src=".\Test\Images\1__refocus_04.jpg" alt="Image 5" style="width: 100px;"></td>
        <td><img src=".\Test\Images\1__refocus_05.jpg" alt="Image 6" style="width: 100px;"></td>
    </tr>
    <tr>
        <td><img src=".\Test\Images\1__refocus_06.jpg" alt="Image 7" style="width: 100px;"></td>
        <td><img src=".\Test\Images\1__refocus_07.jpg" alt="Image 8" style="width: 100px;"></td>
        <td><img src=".\Test\Images\1__refocus_08.jpg" alt="Image 9" style="width: 100px;"></td>
    </tr>
  </table>
</div>

Replace image1.jpg, image2.jpg, ..., image9.jpg, resultant_image1.jpg, and resultant_image2.jpg with the actual file paths or URLs of your images. The alt attribute is used for image descriptions for accessibility.

In this example, rowspan="3" is used to make resultant_image1.jpg span across three rows, and colspan="3" is used to make an empty cell in the last row to align resultant_image2.jpg.

Keep in mind that rendering might vary across different Markdown parsers and viewers. GitHub's Markdown parser supports HTML in README files, so this should work there.







### Contributing

Contributions to this repository are welcome! Feel free to open issues or pull requests for any improvements or additional features.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

