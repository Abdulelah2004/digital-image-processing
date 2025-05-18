# Import the required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image from local file
image = cv2.imread("mypic.jpg")  # Make sure 'mypic.jpg' is in the same directory as this script

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load the image. Make sure 'mypic.jpg' exists in this folder.")
    exit()

# Convert the image from BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Display the original image
plt.figure(figsize=(8, 6))
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# Step 3: Translation (Shift the image)
def translate_image(image, x_shift, y_shift):
    # Create translation matrix
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return translated_image

translated_image = translate_image(image_rgb, 50, 30)  # Shift 50 pixels right, 30 pixels down

# Display translated image
plt.figure(figsize=(8, 6))
plt.title("Translated Image")
plt.imshow(translated_image)
plt.axis('off')
plt.show()

# Step 4: Rotation
(h, w) = image_rgb.shape[:2]
center = (w // 2, h // 2)  # Center point of rotation
angle = 45  # Rotation angle in degrees
scale = 1.0  # Scaling factor

# Create rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (w, h))

# Display rotated image
plt.figure(figsize=(8, 6))
plt.title("Rotated Image")
plt.imshow(rotated_image)
plt.axis('off')
plt.show()

# Step 5: Flipping
flipped_horizontal = cv2.flip(image_rgb, 1)  # Flip horizontally
flipped_vertical = cv2.flip(image_rgb, 0)    # Flip vertically

# Display flipped images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Horizontally Flipped Image")
plt.imshow(flipped_horizontal)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Vertically Flipped Image")
plt.imshow(flipped_vertical)
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 6: Scaling (Resizing)
new_size = (int(w * 0.7), int(h * 0.7))  # Resize to 70% of original size
scaled_image = cv2.resize(image_rgb, new_size)

# Display scaled image
plt.figure(figsize=(8, 6))
plt.title("Scaled Image")
plt.imshow(scaled_image)
plt.axis('off')
plt.show()

print("Process completed! You have performed geometric transformations on the image.")
