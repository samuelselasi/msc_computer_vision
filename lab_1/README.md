# Assignment 1 - Filtering

### 1. Add noise to some images
### 2. Remove the noise using different filters
### 3. Load image, detect the edges using `Robert`, `Sobel`, `Prewitt`, `Canny` and `2nd Derivative` filters

# Introduction

This lab outlines the procedures and results 
of an image processing task focused on noise addition, 
filtering (denoising), and edge detection. 

The goal is to observe how different types of noise affect images, 
how various filters can clean the noise, 
and how edge detectors respond to both clean and noisy data. 

The experiment uses a grayscale image named [input.jpeg](./input.jpeg) 
processed in a Python environment using Ubuntu 20.4


Original Image:
![input](https://github.com/user-attachments/assets/41a7f0d9-f507-47cc-9755-a0d98bc40772)



Grayscale Image:
![Grayscale Image_screenshot_22 06 2025](https://github.com/user-attachments/assets/46525316-67ef-41d8-b3c7-fe1cadb8d44c)


# Methodology

## Adding Noise to the Image

Two common types of noise were artificially added to the image:

1. **Salt-and-Pepper Noise**: Introduces random black and white 
pixels to simulate dead pixels or transmission errors.

2. **Gaussian Noise**: Adds small random variations in pixel 
intensity to simulate sensor noise or low lighting.

Noise Images:
![snp gaussian-noise](https://github.com/user-attachments/assets/2ee56def-c5cd-4e33-a1ba-260b5f6e19c7)


## Denoising Using Filters

To clean the image from the applied noise, 
three different filters were tested:

1. **Median Filter**: Especially effective for `salt-and-pepper` noise.

2. **Gaussian Filter**: Best suited for Gaussian noise.

3. **Mean (Averaging) Filter**: Provides basic smoothing but may blur image details.

Denoising:
![removing-filters](https://github.com/user-attachments/assets/4e05fe43-f858-430e-b760-62ec6684ca20)



# Edge Detection Techniques

1. **Roberts**: A *gradient-based* operator for simple edge detection.

2. **Sobel**: Measures gradients in the `x` and `y` direction.

3. **Prewitt**: Similar to `Sobel`, often used for detecting *vertical* and *horizontal* edges.

4. **Canny**: A *multi-stage* edge detector that includes gradient computation and edge linking.

5. **Laplacian**: Computes the *second derivative* of the image to identify rapid intensity changes.


Edge Detection Techniques:
![edge-detection-techniques](https://github.com/user-attachments/assets/16235d3c-faf3-483b-95ed-d1aba87cff7d)



# Results and Observations

From the filtering experiments, the `Median filter` proved most effective for removing 
`salt-and-pepper noise` while preserving edge details. 

The `Gaussian filter` was more suitable for `Gaussian noise`, 
producing smoother results with less detail loss compared to the Mean filter.

Among the edge detection methods, `Canny` provided the cleanest and 
most connected edges, especially after denoising. 

`Sobel` and `Prewitt` gave similar *gradient-based* results,
while `Laplacian` and `Roberts` highlighted more regions but 
were *more sensitive to noise*.


# Conclusion

This lab demonstrated how various noise types can distort image data
and how specific filters can help restore quality. 

It also highlighted the performance of different edge detectors.

The experiment reinforced the importance of **preprocessing** 
in computer vision tasks and showed how the choice of filter depends on the `noise type`.


