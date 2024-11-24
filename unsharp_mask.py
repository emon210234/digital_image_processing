import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('1.jpg')

# Convert to grayscale for processing
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray_image, (9, 9), 10.0)

# Perform unsharp masking
sharpened = cv2.addWeighted(gray_image, 1.5, blurred, -0.5, 0)

# Display the blurred and the sharpened images in RGB
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax = axes.ravel()

ax[0].imshow(cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB))
ax[0].set_title('Blurred Image')

ax[1].imshow(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
ax[1].set_title('Sharpened Image')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
