# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image from local file in grayscale
image = cv2.imread("mypic.jpg", cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load 'mypic.jpg'. Please make sure it's in the same folder as the code.")
    exit()

# Step 2: Display the original image
plt.figure(figsize=(8, 6))
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

# Step 3: Low-Pass Filter (Gaussian Blur)
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Display the image after Gaussian Blur
plt.figure(figsize=(8, 6))
plt.title("Image after Low-Pass Filter (Gaussian Blur)")
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')
plt.show()

# Step 4: High-Pass Filter (Edge Detection using Sobel)
# Calculate gradients in X and Y directions
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal gradient
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical gradient

# Combine Sobel X and Y results
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)  # Normalize to [0, 255]

# Display the Sobel edge detection result
plt.figure(figsize=(8, 6))
plt.title("Image after High-Pass Filter (Sobel Edge Detection)")
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')
plt.show()

# Step 5: High-Pass Filter (Edge Detection using Canny)
edges = cv2.Canny(image, threshold1=100, threshold2=200)

# Display the Canny edge detection result
plt.figure(figsize=(8, 6))
plt.title("Image after High-Pass Filter (Canny Edge Detection)")
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()

print("Process completed! Image filtering has been successfully applied.")
