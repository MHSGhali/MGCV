# MGCV [WIP]
This repository contains a Python-based image processing pipeline for performing various operations on images. The pipeline is organized into several modules for different tasks. 

## Getting Started

To run the image processing pipeline, follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have the required Python packages installed. You can install them using the following command:
   ```
   pip install -r requirements.txt
   ```

3. Run the pipeline you want. Paths and parameters are command-line arguments:
    ```
    # Focus stacking: each sub-folder of --image-path is stacked into one image
    python run.py --stack --image-path /path/to/images --save-path /path/to/results \
                  --plot-path /path/to/plots --file-type .jpg

    # Panorama stitching: stitches every image in --save-path
    python run.py --stitch --save-path /path/to/results --file-type .jpg
    ```
    Note that `--stitch` reads *and* writes `--save-path`, so a second run in the same
    folder would pick up `panorama.jpg` as an input. Point it at a copy, or clear the
    output first.
4. Stacked images are written to `--save-path` as `stacked_image_<folder>.jpg`, and the
   panorama as `panorama.jpg`.

### Performance

The masking stage dominates runtime. It is split across processes by default, over both
images and row bands within an image; `--workers 1` forces the serial path and
`--workers N` caps the pool. The per-window convex hull underneath it is computed with
cv2 and rasterised with `skimage.measure.grid_points_in_poly` -- the same function
`convex_hull_image` calls internally -- which is ~4x cheaper per window and produces the
identical mask.

On a 14-core machine, a 4-image 800x600 stack at the default window parameters:

| | time |
|---|---|
| original serial code | ~34 s |
| parallel, `convex_hull_image` | 5.4 s |
| parallel, current | **2.3 s** |
| `--select sharpness` | **0.9 s** |

Panorama inputs are decoded on a thread pool (0.32 s -> 0.08 s for 9 x 12 MP frames);
the rest of the stitch time is inside `cv2.Stitcher`.

`bench/run_bench.py` measures per-stage timings and guards against regressions in output:

```
python bench/run_bench.py --golden   # record reference output
python bench/run_bench.py --check    # assert output is still bit-identical
```

It synthesises a deterministic image stack, so it needs no dataset. Every *optimisation*
leaves `masks` and the stacked image bit-identical. Note that the pipeline as a whole is
**not** bit-identical to commit `bad88c2`: two deliberate bug fixes ride along with the
optimisations, in `distibute_unmasked_regions` (a 0/1 mask was passed to `regionprops`
without labelling, so every unmasked pixel formed one region) and in its use of
`normalize_image` (the filtered/residual pair was normalised jointly, then half of it
discarded).

### Output quality

The default `--select hull` assigns whole convex blobs to a single frame, so only ~64% of
its output pixels come from the frame that is actually sharpest there. `--select sharpness`
instead takes each pixel from whichever frame has the highest locally-smoothed Sobel
energy. Measured against the ideal all-sharp scene on the synthetic bench stack:

| | RMSE vs ideal |
|---|---|
| `--select hull` | 13.4 |
| `--select sharpness` | **9.7** |
| best achievable after registration | 9.2 |
| best achievable on unregistered frames | 7.1 |

`--select sharpness` is essentially at the ceiling the pipeline allows. The gap between
the last two rows is `compensate_focal_distance`: it resamples 3 of the 4 frames through
a near-identity homography, which costs real sharpness on a stack that needs no
registration.

Two caveats before trusting this on your own images. It is validated only on the synthetic
stack, whose focus falls off as a smooth Gaussian, and this repository contains no real
focus-bracketed set to check against (`Panorama/Images` is a panorama sweep). That is why
`hull` remains the default. Record a golden for the new mode with its own output
directory:

```
python bench/run_bench.py --golden --select sharpness --out-dir bench/goldens-sharpness
```

Finally, encoding matters more than it looks: writing the stack as default-quality JPEG
cost more accuracy than the stacking algorithm did. JPEG is now written at quality 95, and
`--save-format png` is lossless.

### Options that trade output for speed

These change the result, so they are off unless asked for:

| flag | effect |
|---|---|
| `--pano-scale 0.5` | shrink panorama inputs before stitching |
| `--pano-compose-resol N` | megapixels `Stitcher` composites at (`-1` = full) |
| `--select sharpness` | the per-pixel focus selector described above |

## Modules

### filters.py

This module contains the `Filters` class, which handles various filtering operations on images. The class initializes with the image path and file type and provides methods for performing edge detection, masking, and stacking operations.

#### Infinity Focus (Focus Stacking)
The goal of this pipeline is to stack images with random focus areas intoi a single hyper-focused image.

#### OpenCV Stitch (Panorama Stitching)
The goal of this pipeline is to stitch overalpping images into a panorama, preferred overal greater than 40 %.

### helper.py

The `Helpers` class in this module offers utility functions for loading images, plotting Gaussian pyramids, and displaying images.

### timing.py

A small context manager that records per-stage wall time into `STAGE_TIMES`, used by the
pipeline and read back by the benchmark harness.

### processing.py

The `ImageProcessing` class in this module provides methods for performing image filtering, thresholding, and other image processing techniques.

### Usage and Results

The `run.py` script demonstrates the usage of the pipeline. It loads images from the specified `--image-path`, applies edge detection using the Sobel filter, performs masking and stacking, and saves the stacked image to `--save-path`.

### Contributing

Contributions to this repository are welcome! Feel free to open issues or pull requests for any improvements or additional features.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

