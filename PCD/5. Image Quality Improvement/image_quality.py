# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image from local file in grayscale mode
image = cv2.imread("mypic.jpg", cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load 'mypic.jpg'. Make sure the file exists in the same folder.")
    exit()

# Step 2: Display the original image
plt.figure(figsize=(8, 6))
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

# Step 3: Brightness and Contrast Adjustment
def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    Adjust brightness and contrast.
    alpha: contrast factor (1.0 = no change)
    beta: brightness offset (-127 to +127)
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

# Increase contrast and brightness
brightened_image = adjust_brightness_contrast(image, alpha=1.5, beta=50)

# Display the adjusted image
plt.figure(figsize=(8, 6))
plt.title("Brightness and Contrast Adjusted Image")
plt.imshow(brightened_image, cmap='gray')
plt.axis('off')
plt.show()

# Step 4: Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# Display the equalized image
plt.figure(figsize=(8, 6))
plt.title("Histogram Equalized Image")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')
plt.show()

# Step 5: Filtering (Gaussian Blur and Sharpening)

# Apply Gaussian Blur (Low-pass filter)
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Sharpening (High-pass filter)
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)

# Display blurred and sharpened images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Gaussian Blurred Image")
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Sharpened Image")
plt.imshow(sharpened_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

print("Process completed! Image quality enhancement operations have been applied.")
