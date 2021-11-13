import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage


def part1_q1_3():
    # Create a while board with 200 x 200 and draw a 100 x 100 black square in the middle
    m = np.full((200, 200), 255.0)
    m[51: 151, 51: 151] = 0
    sigma_list = np.linspace(1, 10, 100)

    # Apply Normalized LoG to m using different sigma and save the max response
    response_list = []
    for sigma in sigma_list:
        result_matrix = sigma * sigma * \
            ndimage.gaussian_laplace(m, sigma=sigma)
        max_response = max(abs(np.max(result_matrix)),
                           abs(np.min(result_matrix)))
        response_list.append(max_response)
    max_sigma = sigma_list[response_list.index(max(response_list))]
    # Plot the reuslt
    print(f'The max response occurs when sigma = {max_sigma: .3f}')
    plt.plot(sigma_list, response_list)
    plt.xlabel('Sigma values')
    plt.ylabel('max response')
    plt.show()


def part2_q1(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    print(mag)
    plt.imshow(mag)
    plt.show()


if __name__ == '__main__':
    part1_q1_3()
    # image = plt.imread('./Q3/1.jpg')
    # part2_q1(image)
