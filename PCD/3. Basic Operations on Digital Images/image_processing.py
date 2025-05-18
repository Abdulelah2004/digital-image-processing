# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image from your local directory
image_path = 'mypic.jpg'  # Your local image file
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not found. Make sure 'mypic.jpg' is in the same folder as this script.")
    exit()

# Convert the image from BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Display the original image
plt.figure(figsize=(8, 6))
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# Step 3: Cropping the image
# (y1:y2, x1:x2) — adjust these values as needed
cropped_image = image_rgb[50:200, 50:300]

plt.figure(figsize=(8, 6))
plt.title("Cropped Image")
plt.imshow(cropped_image)
plt.axis('off')
plt.show()

# Step 4: Resize the image
new_size = (300, 300)  # (width, height)
resized_image = cv2.resize(image_rgb, new_size)

plt.figure(figsize=(8, 6))
plt.title("Resized Image")
plt.imshow(resized_image)
plt.axis('off')
plt.show()

# Step 5: Flip the image
flipped_horizontal = cv2.flip(image_rgb, 1)  # Horizontal flip
flipped_vertical = cv2.flip(image_rgb, 0)    # Vertical flip

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Flipped Horizontal")
plt.imshow(flipped_horizontal)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Flipped Vertical")
plt.imshow(flipped_vertical)
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 6: Rotate the image
(h, w) = image_rgb.shape[:2]
center = (w // 2, h // 2)  # Rotation center
angle = 45  # Rotation angle in degrees
scale = 1.0  # Scale factor

rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (w, h))

plt.figure(figsize=(8, 6))
plt.title("Rotated Image")
plt.imshow(rotated_image)
plt.axis('off')
plt.show()

# Step 7: Save processed images to files
cv2.imwrite("cropped_image.jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
cv2.imwrite("resized_image.jpg", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
cv2.imwrite("rotated_image.jpg", cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR))

print("Process complete! Processed images have been saved.")
