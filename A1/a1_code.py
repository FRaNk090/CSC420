import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def gaussian_filter(size: int, sig: float):
    # Calculate the value for the left most and right most position and
    # using linspace to create the coresponding vector
    # for ex, 5 -> [-2, -1, 0, 1, 2]
    # 6 -> [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    right_end = (size - 1) / 2.0
    left_end = -right_end
    # The values in x are values that are going to be sub in gaussian formula
    x = np.linspace(left_end, right_end, size)

    # applying gaussian formula to all entries in x
    gaussian = np.exp(-0.5 * np.square(x) / np.square(sig))

    # taking product of 1d gaussian vector will give us 2D gaussian filter matrix
    gaussian_2d = np.outer(gaussian, gaussian)

    # normalize the entries
    gaussian_2d /= gaussian_2d.sum()
    return gaussian_2d


def convolution(image, filter):
    assert filter.shape[0] == filter.shape[1], 'filter has to be k by k'
    # flip the fiter first
    filter = np.flip(filter)
    k = filter.shape[0]
    m, n = image.shape
    # Since I'm doing valid option here, so the result img size will be smaller
    m = m - k + 1
    n = n - k + 1
    res = np.zeros((m, n))
    # loop through each entry to calculate the result
    for i in range(m):
        for j in range(n):
            res[i][j] = np.sum(image[i:i + k, j:j + k] * filter)
    return res


def gradient_magnitude(image):
    # input image is already in gray scale
    # define sober filter
    sober_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sober_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # using covolution with sober matrix to get derivatives.
    g_x = convolution(image, sober_x)
    g_y = convolution(image, sober_y)
    # Calculate gradient matrix
    res = np.sqrt(g_x ** 2 + g_y ** 2)
    return res


def threshold(image, epsilon):
    # calcutate the matrix mean
    t = image.mean()
    # initial value of t_{i - 1} should be inf so the first iteration will start
    t_last = float('inf')
    while abs(t - t_last) > epsilon:
        t_last = t
        # calcutate low average and high average with last t value
        low_avg = image.mean(where=image < t_last)
        high_avg = image.mean(where=image >= t_last)
        t = (low_avg + high_avg) / 2
    # map every entry to 0 or 255 based on the condition
    mapping_func = np.vectorize(lambda x: 255 if x >= t else 0)
    res = mapping_func(image)
    return res


if __name__ == '__main__':
    SIZE_1 = 10
    SIGMA_1 = 3
    SIZE_2 = 25
    SIGMA_2 = 5
    EPSILON = 0.0001
    gaussian = gaussian_filter(SIZE_1, SIGMA_1)
    gaussian_2 = gaussian_filter(SIZE_2, SIGMA_2)

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)

    fig.add_subplot(4, 4, 1)
    plt.imshow(gaussian, cmap='gray')
    plt.colorbar()
    plt.title('Gaussian Filter')

    fig.add_subplot(4, 4, 3)
    plt.imshow(gaussian_2, cmap='gray')
    plt.colorbar()
    plt.title('Gaussian Filter')

    for i in range(1, 4):
        ax = [None for _ in range(4)]
        img = Image.open(f'./A1_images/image{i}.jpg')
        img = np.array(img)

        print(f'The shape of original image {i} is ', img.shape)
        # Creating original image
        ax[0] = fig.add_subplot(4, 4, 4 * i + 1)
        ax[0].imshow(img)
        ax[0].title.set_text('original image')
        ax[0].set_xticks([]), ax[0].set_yticks([])

        # Convert it to gray scale and apply gaussian blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = convolution(gray, gaussian)
        ax[1] = fig.add_subplot(4, 4, 4 * i + 2)
        ax[1].imshow(img, cmap='gray')
        ax[1].title.set_text('After gaussian blur')
        ax[1].set_xticks([]), ax[1].set_yticks([])
        print(
            f'The shape of image {i} after applying gaussian blur is ', img.shape)

        # Calculating gradient magnitude
        img = gradient_magnitude(img)
        ax[2] = fig.add_subplot(4, 4, 4 * i + 3)
        ax[2].imshow(img, cmap='gray')
        ax[2].title.set_text('Gradient magnitude')
        ax[2].set_xticks([]), ax[2].set_yticks([])
        print(
            f'The shape of image {i} after taking gradient magnitude is ', img.shape)

        # Apply the threshold
        img = threshold(img, EPSILON)
        ax[3] = fig.add_subplot(4, 4, 4 * i + 4)
        ax[3].imshow(img, cmap='gray')
        ax[3].title.set_text('After thresholding')
        ax[3].set_xticks([]), ax[3].set_yticks([])
        print(
            f'The shape of image {i} after applying threshold is ', img.shape)
    plt.show()
