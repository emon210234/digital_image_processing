import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('1.jpg', 0)

# Obtain number of rows and columns of the image
m, n = img.shape

# Develop Averaging filter (3, 3) mask
mask = np.ones([3, 3], dtype=int)
mask = mask / 9

# Convolve the 3x3 mask over the image
img_new = np.zeros([m, n])

for i in range(1, m - 1):
    for j in range(1, n - 1):
        temp = img[i - 1, j - 1] * mask[0, 0] + img[i - 1, j] * mask[0, 1] + img[i - 1, j + 1] * mask[0, 2] \
             + img[i, j - 1] * mask[1, 0] + img[i, j] * mask[1, 1] + img[i, j + 1] * mask[1, 2] \
             + img[i + 1, j - 1] * mask[2, 0] + img[i + 1, j] * mask[2, 1] + img[i + 1, j + 1] * mask[2, 2]
        img_new[i, j] = temp

img_new = img_new.astype(np.uint8)

# Display the original and blurred images side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax = axes.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')

ax[1].imshow(img_new, cmap='gray')
ax[1].set_title('Blurred Image')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
