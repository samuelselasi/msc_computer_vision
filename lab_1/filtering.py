import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage import data, img_as_ubyte
from skimage.filters import gaussian
from scipy.ndimage import median_filter, uniform_filter
from skimage.filters import sobel, prewitt, roberts, laplace
from skimage.feature import canny


# Load the image
image = cv2.imread('input.jpg')

# Check if image was loaded successfully
if image is None:
    print("Error: Image not found or unable to load.")
    exit()

# Display the original image
cv2.imshow('Original Image', image)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)


# ADD NOISE TO IMAGE

# Salt-and-Pepper Noise
sp_noise = random_noise(gray_image, mode='s&p', amount=0.05)
sp_noise = img_as_ubyte(sp_noise)

# Gaussian Noise
gauss_noise = random_noise(gray_image, mode='gaussian', var=0.01)
gauss_noise = img_as_ubyte(gauss_noise)

# Show Noisy Images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(sp_noise, cmap='gray')
axs[0].set_title("Salt & Pepper Noise")
axs[0].axis("off")

axs[1].imshow(gauss_noise, cmap='gray')
axs[1].set_title("Gaussian Noise")
axs[1].axis("off")
plt.tight_layout()
plt.show()

# DENOISE IMAGES USING DIFFERENT FILTERS

# Median Filter
sp_median = median_filter(sp_noise, size=3)
gauss_median = median_filter(gauss_noise, size=3)

# Gaussian Filter
sp_gaussian = gaussian(sp_noise, sigma=1)
gauss_gaussian = gaussian(gauss_noise, sigma=1)

# Mean (Averaging) Filter
sp_mean = uniform_filter(sp_noise, size=3)
gauss_mean = uniform_filter(gauss_noise, size=3)

# Show Denoising Results
titles = ['Original', 'Median', 'Gaussian', 'Mean']
images = [sp_noise, sp_median, sp_gaussian, sp_mean]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title("S&P: " + titles[i])
    plt.axis("off")

images = [gauss_noise, gauss_median, gauss_gaussian, gauss_mean]
for i in range(4):
    plt.subplot(2, 4, i+5)
    plt.imshow(images[i], cmap='gray')
    plt.title("Gaussian: " + titles[i])
    plt.axis("off")
plt.tight_layout()
plt.show()

# EDGE DETECTION

# Roberts
roberts_edges = roberts(gray_image)

# Sobel
sobel_edges = sobel(gray_image)

# Prewitt
prewitt_edges = prewitt(gray_image)

# Canny
canny_edges = canny(gray_image, sigma=1)

# Laplacian (2nd Derivative)
laplacian_edges = laplace(gray_image)

# Plot All Edges
methods = ["Roberts", "Sobel", "Prewitt", "Canny", "Laplacian"]
edges = [roberts_edges, sobel_edges, prewitt_edges, canny_edges, laplacian_edges]

plt.figure(figsize=(15, 6))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(edges[i], cmap='gray')
    plt.title(methods[i])
    plt.axis("off")
plt.tight_layout()
plt.show()
