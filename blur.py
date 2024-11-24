import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to apply smoothing filter (convolution)
def apply_smoothing_filter(a_image, a_kernel_size):
    # Get image dimensions
    img_h, img_w, img_c = a_image.shape

    # Create a smoothing kernel (uniform kernel normalized to 1)
    kernel = np.ones((a_kernel_size, a_kernel_size), np.float32) / (a_kernel_size ** 2)

    # Pad the image with zeros
    pad_size = a_kernel_size // 2
    padded_image = np.pad(a_image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')

    # Initialize the output image
    output = np.zeros_like(a_image)

    # Apply the smoothing filter
    for y in range(img_h):
        for x in range(img_w):
            for c in range(img_c):  # Iterate over color channels
                region = padded_image[y:y + a_kernel_size, x:x + a_kernel_size, c]
                output[y, x, c] = np.sum(region * kernel)

    return output


# Reading the image
image = cv2.imread('1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for displaying with Matplotlib

# Kernel size for smoothing filter
kernel_size = 15  # Larger size for intensified blur

# Applying the smoothing filter
blurred_image = apply_smoothing_filter(image, kernel_size)

# Plotting the original and blurred images
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Blurred Image
plt.subplot(1, 2, 2)
plt.imshow(blurred_image)
plt.title(f'Blurred Image')
plt.axis('off')

plt.tight_layout()
plt.show()
