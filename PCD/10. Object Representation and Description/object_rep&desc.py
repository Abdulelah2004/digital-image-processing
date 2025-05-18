# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image from the local directory in grayscale
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

# Step 3: Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored contour drawing
cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)  # Green contours

# Display the image with contours
plt.figure(figsize=(8, 6))
plt.title("Image with Contours")
plt.imshow(output_image)
plt.axis('off')
plt.show()

# Step 4: Calculate geometric features for each contour
for i, contour in enumerate(contours):
    # Calculate area
    area = cv2.contourArea(contour)
    
    # Calculate perimeter
    perimeter = cv2.arcLength(contour, closed=True)
    
    # Calculate moments and centroid
    moments = cv2.moments(contour)
    centroid_x = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 0
    centroid_y = int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else 0
    
    print(f"Object {i+1}:")
    print(f"  Area: {area}")
    print(f"  Perimeter: {perimeter}")
    print(f"  Centroid: ({centroid_x}, {centroid_y})")
    print("-" * 30)

print("Process complete! You have successfully calculated geometric features for each object in the image.")
