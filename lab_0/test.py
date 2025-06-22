import cv2

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

# Convert back to rgb
rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
cv2.imshow('RGB Image', rgb_image)


# Apply Canny edge detection
edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
cv2.imshow('Edge Detected Image', edges)

# Wait until a key is pressed and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
