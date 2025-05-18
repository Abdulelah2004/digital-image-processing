# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image from local file
image = cv2.imread("mypic.jpg")

# Check if the image was loaded successfully
if image is None:
    print("Image not found. Please ensure the file is in the same folder as this script.")
    exit()

# Convert the image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Display the original image
plt.figure(figsize=(8, 6))
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# Step 3: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.figure(figsize=(8, 6))
plt.title("Grayscale Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')
plt.show()

# Step 4: Resize the image
new_size = (300, 300)
resized_image = cv2.resize(image_rgb, new_size)

# Display the resized image
plt.figure(figsize=(8, 6))
plt.title("Resized Image")
plt.imshow(resized_image)
plt.axis('off')
plt.show()

# Step 5: Save the processed image
cv2.imwrite("processed_image.jpg", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))

print("Processing complete! The processed image has been saved as 'processed_image.jpg'.")
