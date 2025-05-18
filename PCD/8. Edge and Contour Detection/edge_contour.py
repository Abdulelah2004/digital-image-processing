# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image in grayscale mode
image = cv2.imread("mypic.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: 'mypic.jpg' was not found in the directory.")
    exit()

# Step 2: Display the original image
plt.figure(figsize=(8, 6))
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

# Step 3: Edge Detection using Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal gradient
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical gradient

# Combine the horizontal and vertical gradients
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)  # Normalize to [0, 255]

# Display the result of Sobel edge detection
plt.figure(figsize=(8, 6))
plt.title("Edge Detection using Sobel")
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')
plt.show()

# Step 4: Edge Detection using Canny
edges_canny = cv2.Canny(image, threshold1=100, threshold2=200)

# Display the result of Canny edge detection
plt.figure(figsize=(8, 6))
plt.title("Edge Detection using Canny")
plt.imshow(edges_canny, cmap='gray')
plt.axis('off')
plt.show()

# Step 5: Edge Detection using Laplacian
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))  # Take the absolute value and convert to 8-bit

# Display the result of Laplacian edge detection
plt.figure(figsize=(8, 6))
plt.title("Edge Detection using Laplacian")
plt.imshow(laplacian, cmap='gray')
plt.axis('off')
plt.show()

print("Process completed! You have successfully performed edge detection using Sobel, Canny, and Laplacian.")
