import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
    
    height, width = img.shape

    pad_size = filter_size // 2
    padded_img = np.zeros((height + 2 * pad_size, width + 2 * pad_size), dtype=img.dtype)
    padded_img[pad_size:pad_size + height, pad_size:pad_size + width] = img

    # the top and bottom borders
    padded_img[:pad_size, pad_size:pad_size + width] = img[0, :]
    padded_img[pad_size + height:, pad_size:pad_size + width] = img[height - 1, :]

    # the left and right borders
    padded_img[:, :pad_size] = padded_img[:, pad_size:pad_size + 1]
    padded_img[:, pad_size + width:] = padded_img[:, pad_size + width - 1:pad_size + width]

    return padded_img

def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """

    padded_img = padding_img(img, filter_size)

    smoothed_img = np.zeros_like(img)
    pad_size = filter_size // 2

    for i in range(pad_size, padded_img.shape[0] - pad_size):
        for j in range(pad_size, padded_img.shape[1] - pad_size):
            smoothed_img[i - pad_size, j - pad_size] = np.mean(padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1])

    return smoothed_img
    

def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
    padded_img = padding_img(img, filter_size)

    smoothed_img = np.zeros_like(img)
    pad_size = filter_size // 2

    for i in range(pad_size, padded_img.shape[0] - pad_size):
        for j in range(pad_size, padded_img.shape[1] - pad_size):
            smoothed_img[i - pad_size, j - pad_size] = np.median(padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1])

    return smoothed_img


def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    gt_img = gt_img.astype(np.float32)
    smooth_img = smooth_img.astype(np.float32)

    mse = np.mean((gt_img - smooth_img) ** 2)
    psnr_score = 20 * np.log10(255) - 10 * np.log10(mse)

    return psnr_score

def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image)

def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "ex1_images/noise.png" # <- need to specify the path to the noise image
    img_gt = "ex1_images/ori_img.png" # <- need to specify the path to the gt image
    img = read_img(img_noise)
    img_gt = read_img(img_gt)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img_gt, mean_smoothed_img))
    save_image(mean_smoothed_img, "ex1_images/mean_img.png")

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img_gt, median_smoothed_img))
    save_image(mean_smoothed_img, "ex1_images/median_img.png")

    # PSNR score of mean filter:  26.20239307858877
    # PSNR score of median filter:  36.97746088483267