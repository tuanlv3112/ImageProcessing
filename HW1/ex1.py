import cv2 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# Load an image from file as function
def load_image(image_path):
    img = cv2.imread(image_path)
    return img

# Display an image as function
def display_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# grayscale an image as function
def grayscale_image(image):
    height, width, _ = image.shape

    img_gray = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]

            p = 0.299 * r + 0.587 * g + 0.114 * b

            img_gray[y, x] = p

    return img_gray

# Save an image as function
def save_image(image, output_path):
    cv2.imwrite(output_path, image)


# flip an image as function 
def flip_image(image):
    img_flip = cv2.flip(image, 0)
    return img_flip


# rotate an image as function
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


if __name__ == "__main__":
    # Load an image from file
    image_path = '/Users/tuanlv/ImageProcessing/HW1/uet.png'
    img = load_image(image_path)

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")
    save_image(img_gray_flipped, "lena_gray_flipped.jpg")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "lena_gray_rotated.jpg")

    # Show the images
    plt.show() 
