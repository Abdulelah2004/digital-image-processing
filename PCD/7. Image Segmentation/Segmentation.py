# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image from a local file
image = cv2.imread("mypic.jpg")
if image is None:
    print("Error: 'mypic.jpg' not found. Please make sure it's in the same directory as this script.")
    exit()

# Convert the image from BGR (OpenCV format) to RGB (matplotlib format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Display the original image
plt.figure(figsize=(8, 6))
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# Step 3: Color-Based Segmentation
# Convert image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for segmentation (example: red color)
lower_red = np.array([0, 100, 100])   # Lower bound for red
upper_red = np.array([10, 255, 255])  # Upper bound for red

# Create a mask for the red color range
mask = cv2.inRange(hsv_image, lower_red, upper_red)

# Apply the mask to the original RGB image
segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Display the result of color-based segmentation
plt.figure(figsize=(8, 6))
plt.title("Color-Based Segmentation (Red)")
plt.imshow(segmented_image)
plt.axis('off')
plt.show()

# Step 4: Thresholding-Based Segmentation
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Display the result of thresholding segmentation
plt.figure(figsize=(8, 6))
plt.title("Thresholding-Based Segmentation")
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.show()

# Step 5: Edge-Based Segmentation (Canny Edge Detection)
edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

# Display the result of edge detection
plt.figure(figsize=(8, 6))
plt.title("Edge-Based Segmentation (Canny)")
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()

print("Process completed! Image segmentation was successful.")
