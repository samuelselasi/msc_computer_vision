from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

# === STEP 1: Load and transform the image ===
img = cv2.imread('original.jpg')
if img is None:
    raise FileNotFoundError("original.jpg not found in the working directory.")

rows, cols = img.shape[:2]

# 1. Rotation + Scaling
angle = 30
scale = 1.2
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
transformed = cv2.warpAffine(img, M, (cols, rows))

# 2. Viewpoint (Perspective Transform)
pts1 = np.float32([[50, 50], [cols - 50, 50], [50, rows - 50], [cols - 50, rows - 50]])
pts2 = np.float32([[30, 70], [cols - 30, 20], [70, rows - 20], [cols - 60, rows - 80]])
M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
transformed = cv2.warpPerspective(transformed, M_perspective, (cols, rows))

# 3. Illumination (Darken image)
transformed = cv2.convertScaleAbs(transformed, alpha=0.8, beta=-40)

# Save transformed image
cv2.imwrite('transformed.jpg', transformed)

# === STEP 2: Load both images in grayscale ===
img1 = cv2.imread('original.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('transformed.jpg', cv2.IMREAD_GRAYSCALE)

# === STEP 3: Feature detection and description using SURF ===
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)  # Lower = more keypoints
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

# === STEP 4: Descriptor matching using BFMatcher ===
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

if len(matches) == 0:
    raise ValueError("No matches found between the images.")

# === STEP 5: Visualization ===

# === VISUALIZE: Color and Grayscale Comparisons ===
# A. Original vs Transformed (Color)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original (Color)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
plt.title("Transformed (Color)")
plt.axis('off')
plt.tight_layout()
plt.show()

# B. Grayscale Original vs Transformed
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title("Original (Grayscale)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title("Transformed (Grayscale)")
plt.axis('off')
plt.tight_layout()
plt.show()


# A. Show original and transformed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title("Transformed Image (Rot+Scale+View+Illum)")
plt.axis('off')
plt.tight_layout()
plt.show()

# B. Show keypoints
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1_kp)
plt.title("Keypoints in Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2_kp)
plt.title("Keypoints in Transformed")
plt.axis('off')
plt.tight_layout()
plt.show()

# C. Show matched features
matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12, 6))
plt.imshow(matched_image)
plt.title('SURF Feature Matching (Top 50 Matches)')
plt.axis('off')
plt.show()
