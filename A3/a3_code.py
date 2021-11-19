from os import times
import cv2
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import ndimage


def part1_q1_3():
    # Create a while board with 200 x 200 and draw a 100 x 100 black square in the middle
    m = np.full((200, 200), 255.0)
    m[51: 151, 51: 151] = 0
    sigma_list = np.linspace(0.0001, 1, 500)

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
    plt.title('Max response VS sigma')
    plt.xlabel('Sigma values')
    plt.ylabel('max response')
    plt.show()


def retriving_mag_and_angle(image, t):
    # Calculate derivatives
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    # Calculate the magnitude and angle for sourc.e image
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # Convert direct angle to indirect angle with range of -15 ~ 165

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
    threshold = np.vectorize(lambda x: x if x >= t else 0)
    mag = threshold(mag)
    return mag, angle


def create_grid(image, size, mag, angle):
    # crop the image, gradient abd angle based on grid size
    m, n = image.shape
    new_m = m - m % size
    new_n = n - n % size
    image = image[:new_m, :new_n]
    mag = mag[:new_m, :new_n]
    angle = angle[:new_m, :new_n]
    return image, mag, angle, int(new_m / size), int(new_n / size)


def process_cell(angle, mag, mode):
    # Define 6 if function for each bin
    bin_conditions = [(lambda i: lambda x: True if x >= -15 +
                       i * 30 and x < 15 + i * 30 else False)(i) for i in range(6)]
    # Define histogram list
    histogram = [0 for _ in range(6)]
    # Function for processing each pixel

    def process_pixel(x, y, mode):
        for m, bin_condition in enumerate(bin_conditions):
            if bin_condition(x):
                # Accumulate either magnitude or occurrence
                if mode == 'magnitude':
                    histogram[m] += y
                elif mode == 'occurrence' and y != 0:
                    histogram[m] += 1
                break
    process_pixel = np.vectorize(process_pixel, otypes=[])
    process_pixel(angle, mag, mode)
    return histogram


def normalize_descriptor(descriptor):
    m, n, _ = descriptor.shape
    normalized_descriptor = np.zeros((m - 1, n - 1, 24))
    for i in range(m - 1):
        for j in range(n - 1):
            # putting 4 descriptors together into 24 x 1 vector
            result = np.concatenate([descriptor[i][j],
                                    descriptor[i][j + 1],
                                    descriptor[i + 1][j],
                                    descriptor[i + 1][j + 1]])
            # Normalize descriptor and store in matrix
            normalized_descriptor[i][j] = result / \
                np.sum(np.sqrt(np.square(result) + np.square(0.001)))
    return normalized_descriptor


def second_moment_matrix(image, sigma):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    m, n = gray.shape
    # Get Ix and Iy
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)

    # Kernel size can be defined here
    k_size = 11
    Ix2_blur = cv2.GaussianBlur(Ix2, (k_size, k_size), sigma)
    Iy2_blur = cv2.GaussianBlur(Iy2, (k_size, k_size), sigma)
    IxIy_blur = cv2.GaussianBlur(IxIy, (k_size, k_size), sigma)
    # Create Eigen value matrix
    eigen_values = np.zeros((m, n, 2))

    for i in range(m):
        for j in range(n):
            eigen_values[i][j] = LA.eigvals(np.array([[Ix2_blur[i][j], IxIy_blur[i][j]],
                                                     [IxIy_blur[i][j], Iy2_blur[i][j]]]))

    return eigen_values


def calculate_quiver(descriptor):
    x_pos, y_pos, x_direct, y_direct = [], [], [], []
    m, n, _ = descriptor.shape
    for i in range(m):
        for j in range(n):
            for m in range(6):
                # Only draw lines when magnitude is > 0
                if descriptor[i][j][m] > 0:
                    y_pos.append(i * CELL_LENGTH + np.ceil(CELL_LENGTH / 2.0))

                    x_pos.append(j * CELL_LENGTH + np.ceil(CELL_LENGTH / 2.0))

                    # Since lines are perpendicular to gradiant magnitude, use sin in x direction and cos in y direction
                    x_direct.append(np.sin(np.pi / 6 * m)
                                    * descriptor[i][j][m])

                    y_direct.append(np.cos(np.pi / 6 * m)
                                    * descriptor[i][j][m])

    return x_pos, y_pos, x_direct, y_direct


def calculate_quiver_normalized(descriptor):
    x_pos, y_pos, x_direct, y_direct = [], [], [], []
    m, n, _ = descriptor.shape
    for i in range(m):
        for j in range(n):
            matrix = descriptor[i][j].reshape((4, 6))
            # Sum 4 vectors into 1
            des_vector = np.sum(matrix, axis=0)
            for m in range(6):
                # Only draw lines when magnitude is > 0
                if des_vector[m] > 0:
                    y_pos.append((i + 1) * CELL_LENGTH)
                    x_pos.append((j + 1) * CELL_LENGTH)
                    # Since lines are perpendicular to gradiant magnitude, use sin in x direction and cos in y direction
                    x_direct.append(np.sin(np.pi / 6 * m) * des_vector[m])
                    y_direct.append(np.cos(np.pi / 6 * m) * des_vector[m])
    return x_pos, y_pos, x_direct, y_direct


if __name__ == '__main__':

    # ===== Part 1 Q1.3
    part1_q1_3()

    CELL_LENGTH = 8
    # Threshold for each images are defined here
    THRESHOLD = [50, 70, 80, 60, 60]

    # ======= Part 2 Q1 - Q3 ==========

    for image_id in range(1, 6):
        image = cv2.imread(f'./Q3/{image_id}.jpg')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mag, angle = retriving_mag_and_angle(gray, THRESHOLD[image_id - 1])
        gray, mag, angle, grid_m, grid_n = create_grid(
            gray, CELL_LENGTH, mag, angle)
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        ax = [None for _ in range(4)]
        # create descriptor matrix for magnitude
        descriptor_mag = np.zeros((grid_m, grid_n, 6))
        for i in range(grid_m):
            for j in range(grid_n):
                # process each cell and save descriptor.
                descriptor_mag[i, j] = process_cell(angle[i * CELL_LENGTH: (i + 1) * CELL_LENGTH,
                                                          j * CELL_LENGTH: (j + 1) * CELL_LENGTH],
                                                    mag[i * CELL_LENGTH: (i + 1) * CELL_LENGTH,
                                                    j * CELL_LENGTH: (j + 1) * CELL_LENGTH],
                                                    mode='magnitude')

        x_pos, y_pos, x_direct, y_direct = calculate_quiver(descriptor_mag)
        ax[0] = fig.add_subplot(2, 2, 1)
        ax[0].set_title('Magnitude')
        ax[0].imshow(gray, cmap='gray')
        ax[0].quiver(x_pos, y_pos, x_direct, y_direct, color='r',
                     headlength=0, headwidth=1, pivot='mid')

        # create descriptor matrix for occurrence
        descriptor_occ = np.zeros((grid_m, grid_n, 6))
        for i in range(grid_m):
            for j in range(grid_n):
                # process each cell and save descriptor.
                descriptor_occ[i, j] = process_cell(angle[i * CELL_LENGTH: (i + 1) * CELL_LENGTH,
                                                          j * CELL_LENGTH: (j + 1) * CELL_LENGTH],
                                                    mag[i * CELL_LENGTH: (i + 1) * CELL_LENGTH,
                                                    j * CELL_LENGTH: (j + 1) * CELL_LENGTH],
                                                    mode='occurrence')

        x_pos, y_pos, x_direct, y_direct = calculate_quiver(descriptor_occ)
        ax[1] = fig.add_subplot(2, 2, 2)
        ax[1].set_title('Occurrence')
        ax[1].imshow(gray, cmap='gray')
        ax[1].quiver(x_pos, y_pos, x_direct, y_direct, color='r',
                     headlength=0, headwidth=1, pivot='mid')

        # Create normalized descriptor for magnitude
        normalized_descriptor_mag = normalize_descriptor(descriptor_mag)
        x_pos, y_pos, x_direct, y_direct = [], [], [], []
        np.savetxt(f'./{image_id}.txt', normalized_descriptor_mag.reshape(
            (normalized_descriptor_mag.shape[0], normalized_descriptor_mag.shape[1] * normalized_descriptor_mag.shape[2])))

        x_pos, y_pos, x_direct, y_direct = calculate_quiver_normalized(
            normalized_descriptor_mag)
        ax[2] = fig.add_subplot(2, 2, 3)
        ax[2].set_title('Normalized Magnitude')
        ax[2].imshow(gray, cmap='gray')
        ax[2].quiver(x_pos, y_pos, x_direct, y_direct, color='r',
                     headlength=0, headwidth=1, pivot='mid')

        plt.show()

    # ======= Part 2 Q4 ==========

    sigma = [2, 9]
    threshold = [13000000, 8000000]
    for image_id in range(6, 8):
        image = cv2.imread(f'./Q3/{image_id}.jpg')
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        ax = [None for _ in range(4)]
        # == Plot for the first sigma value ==
        x_values, y_values = [], []
        # Calculate eigon values
        eigen_values = second_moment_matrix(image, sigma[0])
        m, n, _ = eigen_values.shape
        for i in range(m):
            for j in range(n):
                x_values.append(eigen_values[i][j][0])
                y_values.append(eigen_values[i][j][1])
        # Scatter the eigen values and plot the corner
        ax[0] = fig.add_subplot(2, 2, 1)
        ax[0].scatter(x_values, y_values)
        ax[0].set_title(f'Scatter plot with sigma = {sigma[0]}')
        img_copy = image.copy()
        for i in range(m):
            for j in range(n):
                if min(eigen_values[i][j]) > threshold[image_id - 6]:
                    cv2.circle(img_copy, (j, i), 2, (255, 255, 0), -1)
        ax[1] = fig.add_subplot(2, 2, 2)
        ax[1].imshow(img_copy)
        ax[1].set_title(f'Corners with sigma = {sigma[0]}')

        # == Plot for the second sigma value
        x_values, y_values = [], []
        # Calculate eigon values
        eigen_values = second_moment_matrix(image, sigma[1])
        m, n, _ = eigen_values.shape
        for i in range(m):
            for j in range(n):
                x_values.append(eigen_values[i][j][0])
                y_values.append(eigen_values[i][j][1])
        # Scatter the eigen values and plot the corner
        ax[2] = fig.add_subplot(2, 2, 3)
        ax[2].scatter(x_values, y_values)
        ax[2].set_title(f'Scatter plot with sigma = {sigma[1]}')
        img_copy = image.copy()
        for i in range(m):
            for j in range(n):
                if min(eigen_values[i][j]) > threshold[image_id - 6]:
                    cv2.circle(img_copy, (j, i), 2, (255, 255, 0), -1)
        ax[3] = fig.add_subplot(2, 2, 4)
        ax[3].imshow(img_copy)
        ax[3].set_title(f'Corners with sigma = {sigma[1]}')
        plt.show()
