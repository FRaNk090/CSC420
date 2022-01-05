import cv2
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import mplcursors
import sys

def get_points_selected(gray1):
    '''Return the selected points on two figures
    '''

    # Create figure
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    ax = [None for _ in range(2)]

    ax[0] = fig.add_subplot(1, 1, 1)
    a = ax[0].imshow(gray1, cmap='gray')

    # ax[1] = fig.add_subplot(1, 2, 2)
    # b = ax[1].imshow(gray2, cmap='gray')

    points_1 = []
    # points_2 = []

    # Define call back event when click on fig
    def on_click(sel):
        x = int(sel.target_[0])
        y = int(sel.target_[1])
        # Store x, y coordinates of click points
        if sel.artist == a:
            points_1.append([x, y])
            sel.annotation.set_text(f'x: {x}\ny: {y} \npoint{len(points_1)}')
            # print(gray1[sel.target_[1]][sel.target_[0]])
        # elif sel.artist == b:
        #     points_2.append([x, y])
        #     sel.annotation.set_text(f'x: {x}\ny: {y} \npoint{len(points_2)}')
    mplcursors.cursor(multiple=True).connect("add", on_click)

    plt.show()
    # Make sure that more than 4 pair of points are selectedl otherwise raise assertion error
    # assert len(points_1) == 4, 'You should select 4 points'
    # assert len(points_2) >= len(
    #     points_1), 'You should select the same amount of points on figure 2'
    return points_1

def matrix_for_point(point1, point2):
    '''Given a pair of point, produce a 2 x 9 matrix.
        x, y, 1, 0, 0, 0, -x'x, -x'y, -x'
        0, 0, 0, x, y, 1, -y'x, -y'y, -y'
    '''
    # return the matrix A based on formula
    res = np.array([[point1[0], point1[1], 1, 0, 0, 0, -point2[0] * point1[0], -point2[0] * point1[1], -point2[0]],
                    [0, 0, 0, point1[0], point1[1], 1, -point2[1] * point1[0], -point2[1] * point1[1], -point2[1]]])
    return res

def calculate_homography_matrix(points_1, points_2):
    '''Given two groups of points to estimate h matrix
    '''
    # Make sure that the number of points in 2 groups are the same
    assert len(points_1) == len(points_2), "Number of points must be the same"
    length = len(points_1)
    # Stack the matrix vertically
    A_matrix = np.array([[]]).reshape((0, 9))
    for i in range(length):
        point_matrix = matrix_for_point(points_1[i], points_2[i])
        A_matrix = np.vstack((A_matrix, point_matrix))
    # Get the eigenvector associated with the smallest eigenvalue
    m = np.matmul(A_matrix.T, A_matrix)
    w, v = LA.eig(m)
    smallest_index = np.argmin(np.abs(w))
    # Get the eigenvector of the smallest eigenvalue
    h = v[:, smallest_index].reshape((3, 3))
    return h

def illustrate_effect(shape_1, m):
    h, w = shape_1

    white = np.zeros((h * 2, w * 2), dtype=np.uint8)
    white[250: 250 + h, 500: 500 + w] = 255
    warped = cv2.warpPerspective(white, m, (w * 2, h * 2))
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    ax = [None for _ in range(2)]

    ax[0] = fig.add_subplot(1, 2, 1)
    ax[0].imshow(white, cmap='gray')

    ax[1] = fig.add_subplot(1, 2, 2)
    ax[1].imshow(warped, cmap='gray')
    plt.show()

def homogeneous_transformation(points, h):
    '''Given a list of points, return the points after
        homogeneous transformation
    '''
    result_point = []
    for point in points:
        assert len(point) == 2, "Point has to be defined in 2D"
        # Use formula from slide to calculate the value of (x, y)
        # after homogeneous transformation
        x = (h[0][0] * point[0] + h[0][1] * point[1] + h[0][2]) / \
            (h[2][0] * point[0] + h[2][1] * point[1] + h[2][2])
        y = (h[1][0] * point[0] + h[1][1] * point[1] + h[1][2]) / \
            (h[2][0] * point[0] + h[2][1] * point[1] + h[2][2])
        result_point.append([int(round(x)), int(round(y))])
    return result_point

def image_transformation(image, shape, h):
    height, width = shape

    # result_image = np.zeros((height, width, 3), dtype=np.int32)
    # # Add some offset to image so every point will appear
    # points_cord = [[x, y] for y in range(1, height + 1)
    #                for x in range(1, width + 1)]
    # # print(points_cord)
    # # Store the coordinates after applying homogeneous
    # result_points = np.array(homogeneous_transformation(points_cord, LA.inv(h)))
    # print(result_points)
    # for i, point in enumerate(result_points):
    #     if 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]:
    #         # print(point)
    #         result_image[i // width][i % width] = image[point[1]][point[0]]
            
    # Set the Red channel to be image 1
    # result_image[500: image1.shape[0] + 500,
    #              750: image1.shape[1] + 750, 0] = image1
    result_image = cv2.warpPerspective(image, h, (width, height))

    plt.imshow(result_image)
    plt.show()
    return result_image

if __name__ == '__main__':
    # np.set_printoptions(threshold=sys.maxsize)
    result_shape = (600, 400)

    height, width = result_shape
    image = cv2.imread(f'./images/book.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    points_1 = get_points_selected(image)
    # points_1 = [[739, 333], [1063, 20], [736, 562], [1056, 865]]
    points_2 = [[1, 1], [width, 1], [1, height], [width, height]]
    # print(points_1)
    # test_point = [[10, 10]]
    h = calculate_homography_matrix(points_1, points_2)
    print(homogeneous_transformation(points_1, h))
    # for i in range(1, 21):
    #     for j in range(1, 21):
    #         print(homogeneous_transformation([[j, i]], LA.inv(h)))
    print(homogeneous_transformation(points_2, LA.inv(h)))
    
    image_transformation(image, result_shape, h)