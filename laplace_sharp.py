import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma=1):
    k = cv2.getGaussianKernel(size, sigma)
    return np.outer(k, k)

def apply_convolution(a_image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = a_image.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(a_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output = np.zeros_like(a_image)
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
    return output

def laplacian_operator(l_image):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return apply_convolution(l_image, kernel)

# Load the image
image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
gaussian_blur_kernel = gaussian_kernel(3, 1)
blurred = apply_convolution(image, gaussian_blur_kernel)

# Apply the Laplacian operator
laplacian = laplacian_operator(blurred)

# Convert back to uint8
laplacian = cv2.convertScaleAbs(laplacian)

# Sharpen the image by adding the Laplacian to the original image
sharpened = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
plt.subplot(1, 3, 3), plt.imshow(sharpened, cmap='gray'), plt.title('Sharpened')
plt.show()