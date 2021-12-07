import cv2
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import mplcursors


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
            sel.annotation.set_text(f'x: {x}\ny: {y} \npoint{len(points_2)}')
    mplcursors.cursor(multiple=True).connect("add", on_click)

    plt.show()
    # Make sure that more than 4 pair of points are selectedl otherwise raise assertion error
    assert len(points_1) > 4, 'You should select more than 4 points on figure 1'
    assert len(points_2) >= len(
        points_1), 'You should select the same amount of points on figure 2'
    return points_1, points_2


def show_points(img1, img2, points_1, points_2):
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    ax = [None for _ in range(2)]

    ax[0] = fig.add_subplot(1, 2, 1)
    ax[0].imshow(img1, cmap='gray')
    for point in points_1:
        ax[0].plot(point[0], point[1], 'rs')

    ax[1] = fig.add_subplot(1, 2, 2)
    ax[1].imshow(img2, cmap='gray')
    for point in points_2:
        ax[1].plot(point[0], point[1], 'rs')
    plt.show()


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


def image_transformation(image1, image2, h):
    height, width = image1.shape[0] * 2, image1.shape[1] * 2

    result_image = np.zeros((height, width, 3), dtype=np.int32)
    # Add some offset to image so every point will appear
    points_cord = [[x, y] for y in range(-500, height - 500)
                   for x in range(-750, width - 750)]
    # Store the coordinates after applying homogeneous
    result_points = np.array(homogeneous_transformation(points_cord, h))

    for i, point in enumerate(result_points):
        if 0 <= point[0] < image2.shape[1] and 0 <= point[1] < image2.shape[0]:
            # If in range, set the Green and Blue channel to be image 2
            result_image[i // width][i % width][1] = image2[point[1]][point[0]]
            result_image[i // width][i % width][2] = image2[point[1]][point[0]]
    # Set the Red channel to be image 1
    result_image[500: image1.shape[0] + 500,
                 750: image1.shape[1] + 750, 0] = image1

    plt.imshow(result_image)
    plt.show()
    return result_image


if __name__ == '__main__':

    #  =======  part2 Q4 ==========
    images = {'A': (1, 2), 'B': (1, 3), 'C': (1, 3)}
    # Cases can be defined here
    case = 'C'
    id1, id2 = images[case][0], images[case][1]
    # Read two images based on cases
    image1 = cv2.imread(f'./Q4/hallway{id1}.jpg')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    image2 = cv2.imread(f'./Q4/hallway{id2}.jpg')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    # Uncomment the line 168 and comment lines 169 - 179 to select points manually
    # points_1, points_2 = get_points_selected(gray1, gray2)
    if case == 'A':
        points_1 = [[821, 645], [950, 263], [
            886, 231], [926, 141], [1097, 230]]
        points_2 = [[700, 967], [807, 577], [742, 548], [777, 458], [948, 539]]
    elif case == 'B':
        points_1 = [[1069, 14], [1098, 177], [
            851, 462], [786, 587], [739, 374]]
        points_2 = [[940, 207], [965, 366], [830, 650], [791, 777], [761, 567]]
    elif case == 'C':
        points_1 = [[997, 820], [821, 648], [501, 751], [571, 563], [657, 545]]
        points_2 = [[922, 997], [817, 844], [449, 950], [595, 766], [693, 739]]

    print(f'selected points for figure 1 are {points_1}')
    print(f'selected points for figure 2 are {points_2}')
    show_points(gray1, gray2, points_1, points_2)

    h = calculate_homography_matrix(points_1, points_2)
    print(f'h is {h}')

    result_points = homogeneous_transformation(points_1, h)

    illustrate_effect(gray1.shape, h)

    image2_copy = image2.copy()

    # Draw rectangle on image
    for i in range(len(result_points)):
        x, y = result_points[i][0], result_points[i][1]
        cv2.rectangle(image2_copy, (x - 15, y - 15),
                      (x + 15, y + 15), (0, 255, 0), 3)
        x, y = points_2[i][0], points_2[i][1]
        cv2.rectangle(image2_copy, (x - 10, y - 10),
                      (x + 10, y + 10), (255, 0, 0), 3)
    plt.imshow(image2_copy)
    plt.show()
    result_image = image_transformation(gray1, gray2, h)
