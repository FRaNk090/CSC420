import cv2
from matplotlib.colors import PowerNorm
import numpy as np
from numpy import int32, linalg as LA
import matplotlib.pyplot as plt
from scipy import ndimage
import mplcursors
from numpy import linalg as LA


def get_points_selected(gray1, gray2):
    '''Return the selected points on two figures
    '''

    # Create figure
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    ax = [None for _ in range(2)]

    ax[0] = fig.add_subplot(1, 2, 1)
    a = ax[0].imshow(gray1, cmap='gray')

    ax[1] = fig.add_subplot(1, 2, 2)
    b = ax[1].imshow(gray2, cmap='gray')

    points_1 = []
    points_2 = []
    # Define call back event when click on fig

    def on_click(sel):
        x = int(sel.target_[0])
        y = int(sel.target_[1])
        # Store x, y coordinates of click points
        if sel.artist == a:
            points_1.append([x, y])
            sel.annotation.set_text(f'x: {x}\ny: {y} \npoint{len(points_1)}')
            # print(gray1[sel.target_[1]][sel.target_[0]])
        elif sel.artist == b:
            points_2.append([x, y])
            sel.annotation.set_text(f'x: {x}, y: {y} \npoint{len(points_2)}')
    mplcursors.cursor(multiple=True).connect(
        "add", on_click)

    plt.show()
    # Make sure that more than 4 pair of points are selectedl otherwise raise assertion error
    assert len(points_1) >= 4, 'You should select more than 4 points on figure 1'
    assert len(points_2) >= 4, 'You should select more than 4 points on figure 2'
    return points_1, points_2


def matrix_for_point(point1, point2):
    '''Given a pair of point, produce a 2 x 9 matrix.
        x, y, 1, 0, 0, 0, -x'x, -x'y, -x'
        0, 0, 0, x, y, 1, -y'x, -y'y, -y
    '''
    # return the matrix based on formula
    res = np.array([[point1[0], point1[1], 1, 0, 0, 0, -point2[0] * point1[0], -point2[0] * point1[1], -point1[0]],
                    [0, 0, 0, point1[0], point1[1], 1, -point2[1] * point1[0], -point2[1] * point1[1], -point2[1]]])
    return res


def calculate_homography_matrix(points_1, points_2):
    '''Given two pairs fo
    '''
    # Make sure that the number of points in 2 groups are the same
    assert len(points_1) == len(points_2), "Number of points must be the same"
    length = len(points_1)
    # Stack the matrix
    A_matrix = np.array([[]]).reshape((0, 9))
    for i in range(length):
        point_matrix = matrix_for_point(points_1[i], points_2[i])
        A_matrix = np.vstack((A_matrix, point_matrix))
    # Get the eigenvector associated with the smallest eigenvalue
    m = np.matmul(A_matrix.T, A_matrix)
    w, v = LA.eig(m)
    smallest_index = np.argmin(np.abs(w))
    h = v[:, smallest_index].reshape((3, 3))
    return h


def homogeneous_transformation(points, h):
    '''Given a list of points, return the points after
        homogeneous transformation
    '''
    result_point = []
    for point in points:
        assert len(point) == 2, "Point has to be defined in 2D"
        # Add 1 to make it homogeneous coordinate
        point.append(1)
        point = np.array(point)
        res = np.matmul(h, point)
        result_point.append(np.round((res / res[2])[:2]).astype(int))

    return result_point


def image_transformation(image1, image2, h):
    height, width = image1.shape
    pad = 25
    result_image = np.zeros(
        (height + 2 * pad, width + 2 * pad, 3), dtype=int32)
    for y in range(result_image.shape[0]):
        for x in range(result_image.shape[1]):
            if y in range(pad, result_image.shape[0] - pad) and x in range(pad, result_image.shape[1] - pad):
                result_image[y][x][0] = image1[y - pad][x - pad]
            result_point = homogeneous_transformation([[x, y]], h)[0]
            if result_point[0] in range(image2.shape[1]) and result_point[1] in range(image2.shape[0]):
                result_image[y][x][1] = image2[result_point[1]
                                               ][result_point[0]]
                result_image[y][x][2] = image2[result_point[1]
                                               ][result_point[0]]
            print(result_image[y][x])
    return result_image


if __name__ == '__main__':

    #  =======  part2 Q4 ==========
    images = {'A': (1, 2), 'B': (1, 3), 'C': (1, 3)}
    # Cases can be defined here
    case = 'A'
    id1, id2 = images[case][0], images[case][1]
    # Read two images based on cases
    image1 = cv2.imread(f'./Q4/hallway{id1}.jpg')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    image2 = cv2.imread(f'./Q4/hallway{id2}.jpg')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    points_1, points_2 = get_points_selected(gray1, gray2)
    # points_1 = [[1068, 234], [1071, 438], [1408, 90], [1332, 435]]
    # points_2 = [[925, 520], [944, 750], [1330, 357], [1235, 729]]
    print(points_1, points_2)
    h = calculate_homography_matrix(points_1, points_2)
    result_points = homogeneous_transformation(points_1, h)
    print(result_points)
    image2_copy = image2.copy()
    for i in range(len(result_points)):
        x, y = result_points[i][0], result_points[i][1]
        cv2.rectangle(image2_copy, (x - 10, y - 10),
                      (x + 10, y + 10), (0, 255, 0), 3)
        x, y = points_2[i][0], points_2[i][1]
        cv2.rectangle(image2_copy, (x - 10, y - 10),
                      (x + 10, y + 10), (255, 0, 0), 3)
    plt.imshow(image2_copy)
    plt.show()
    result_image = image_transformation(gray1, gray2, h)
    plt.imshow(result_image)
    plt.show()
