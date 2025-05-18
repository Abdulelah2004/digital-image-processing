# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image in grayscale
image = cv2.imread("mypic.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: 'mypic.jpg' not found in the directory.")
    exit()

# Step 2: Convert the image to binary using thresholding
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Display the binary image
plt.figure(figsize=(8, 6))
plt.title("Binary Image")
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.show()

# Step 3: Define the kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel of ones

# Step 4: Erosion operation
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# Display the result of erosion
plt.figure(figsize=(8, 6))
plt.title("Erosion Result")
plt.imshow(eroded_image, cmap='gray')
plt.axis('off')
plt.show()

# Step 5: Dilation operation
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Display the result of dilation
plt.figure(figsize=(8, 6))
plt.title("Dilation Result")
plt.imshow(dilated_image, cmap='gray')
plt.axis('off')
plt.show()

# Step 6: Opening operation (erosion followed by dilation)
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Display the result of opening
plt.figure(figsize=(8, 6))
plt.title("Opening Result")
plt.imshow(opened_image, cmap='gray')
plt.axis('off')
plt.show()

# Step 7: Closing operation (dilation followed by erosion)
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Display the result of closing
plt.figure(figsize=(8, 6))
plt.title("Closing Result")
plt.imshow(closed_image, cmap='gray')
plt.axis('off')
plt.show()

print("Process completed! You have successfully performed morphological operations on the binary image.")
