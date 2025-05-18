# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

# Step 1: Load image from local file
# Replace 'mypic.jpg' with your own image file name
image = cv2.imread("mypic.jpg")

# Check if the image was loaded successfully
if image is None:
    print("Image not found. Make sure the file is in the same folder as this Python file.")
    exit()

# Convert image from BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Display the original image
plt.figure(figsize=(8, 6))
plt.title("Original Image (RGB)")
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# Step 3: Split the RGB color channels
r, g, b = cv2.split(image_rgb)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Red Channel (R)")
plt.imshow(r, cmap='Reds')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Green Channel (G)")
plt.imshow(g, cmap='Greens')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Blue Channel (B)")
plt.imshow(b, cmap='Blues')
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 4: Convert image to Grayscale
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(8, 6))
plt.title("Grayscale Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')
plt.show()

# Step 5: Convert image to CMYK
def rgb_to_cmyk(rgb_image):
    rgb_normalized = rgb_image.astype(np.float32) / 255.0
    k = 1 - np.max(rgb_normalized, axis=2)
    c = (1 - rgb_normalized[:, :, 0] - k) / (1 - k + 1e-10)
    m = (1 - rgb_normalized[:, :, 1] - k) / (1 - k + 1e-10)
    y = (1 - rgb_normalized[:, :, 2] - k) / (1 - k + 1e-10)
    return c, m, y, k

c, m, y, k = rgb_to_cmyk(image_rgb)

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.title("Cyan Channel (C)")
plt.imshow(c, cmap='Blues')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Magenta Channel (M)")
plt.imshow(m, cmap='Reds')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Yellow Channel (Y)")
plt.imshow(y, cmap='YlOrBr')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Black Channel (K)")
plt.imshow(k, cmap='Greys')
plt.axis('off')

plt.tight_layout()
plt.show()

print("Process completed! You have visualized the RGB, Grayscale, and CMYK color models.")
