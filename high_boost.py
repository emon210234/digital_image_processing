import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

image_path = '1.jpg'

sharpening_factor = 4

def load_image(image_filename):
    try:
        # Load the image
        image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            print(f"Image {image_filename} loaded successfully!")
            return image, os.path.splitext(image_filename)[0]  # Return the image name without extension
        else:
            print(f"Could not load the image {image_filename}. Check the path.")
            return None, None
    except Exception as e:
        print(f"Error loading the image: {str(e)}")
        return None, None

def apply_low_pass_filter(image, kernel_size, sigma):
    offset = kernel_size // 2
    x, y = np.arange(-offset, offset + 1), np.arange(-offset, offset + 1)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

def high_boost_filter(image, h_sharpening_factor, image_name):
    filtered_image = apply_low_pass_filter(image, kernel_size=5, sigma=1)
    sharpened_image = cv2.addWeighted(image, 1 + h_sharpening_factor, filtered_image, -h_sharpening_factor, 0)
    return sharpened_image, image_name

def save_image(image, output_filename):
    output_path = os.path.join('Output_Images', output_filename + '_Sharpened.jpg')
    cv2.imwrite(output_path, image)
    print(f"Image saved at {output_path}")

def main():
    loaded_image, image_name = load_image(image_path)

    if loaded_image is not None:
        sharpened_image, image_name = high_boost_filter(loaded_image, sharpening_factor, image_name)
        save_image(sharpened_image, image_name)

        # Display the images side by side in a matplotlib window
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax = axes.ravel()

        ax[0].imshow(loaded_image, cmap='gray')
        ax[0].set_title('Original Image')

        ax[1].imshow(sharpened_image, cmap='gray')
        ax[1].set_title('Image with Increased Sharpness')

        for a in ax:
            a.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
