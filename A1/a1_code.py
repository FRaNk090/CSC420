import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('./A1_images/image2.jpg')

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


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

    return gaussian_2d / gaussian_2d.sum()


plt.imshow(gaussian_filter(50, 10))
plt.show()
