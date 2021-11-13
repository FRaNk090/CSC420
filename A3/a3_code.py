from os import times
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import sys

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


def retriving_mag_and_angle(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    # Calculate the magnitude and angle for source image
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # Convert direct angle to indirect angle
    def convert_to_indirect_angle(x):
        if x >= 165 and x < 345:
            return x - 180
        elif x >= 345:
            return x - 360
        else:
            return x
    convert_to_indirect_angle = np.vectorize(convert_to_indirect_angle)
    angle = convert_to_indirect_angle(angle)
    # Threshold magnitude
    t = 0
    threshold = np.vectorize(lambda x: x if x >= t else 0)
    mag = threshold(mag)
    # plt.imshow(mag, cmap='gray')
    # plt.show()
    return mag, angle

def create_grid(image, size, mag, angle):
    m, n = image.shape
    new_m = m - m % size
    new_n = n - n % size
    image = image[:new_m, :new_n]
    mag = mag[:new_m, :new_n]
    angle = angle[:new_m, :new_n]
    return image, mag, angle, int(new_m / size), int(new_n / size)

def process_cell(angle, mag, mode):
    # Define 6 if function and histogram list
    bin_conditions = [(lambda i: lambda x : True if x >= -15 + i * 30 and x < 15 + i * 30 else False)(i) for i in range(6)]

    histogram = [0 for i in range(6)]

    times_process = 0
    def process_pixel(x, y, mode):
        for m, bin_condition in enumerate(bin_conditions):
            if bin_condition(x):
                if mode == 'magnitude':
                    histogram[m] += y
                elif mode == 'occurrence':
                    histogram[m] += 1
                break
    process_pixel = np.vectorize(process_pixel, otypes=[])
    process_pixel(angle, mag, mode)
    return histogram
    # for i in range(size):
    #     for j in range(size):

    #         # For each pixel, need to check which bin it belongs to
    #         for m, bin_condition in enumerate(bin_conditions):
    #             if bin_condition(angle[i, j]):
    #                 if mode == 'magnitude':
    #                     histogram[m] += mag[i, j]
    #                 elif mode == 'occurrence':
    #                     histogram[m] += 1

if __name__ == '__main__':
    
    # part1_q1_3()

    # ======= Part 2 Q1 - Q3 ==========
    CELL_LENGTH = 8
    image = cv2.imread('./Q3/1.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mag, angle = retriving_mag_and_angle(gray)
    gray, mag, angle, grid_m, grid_n = create_grid(gray, CELL_LENGTH, mag, angle)
    plt.imshow(mag, cmap='gray')
    for i in range(grid_m):
        for j in range(grid_n):
            print(process_cell(angle[i * CELL_LENGTH : (i + 1) * CELL_LENGTH, j * CELL_LENGTH : (j + 1) * CELL_LENGTH], 
                                mag[i * CELL_LENGTH : (i + 1) * CELL_LENGTH, j * CELL_LENGTH : (j + 1) * CELL_LENGTH],
                                mode='magnitude'))
