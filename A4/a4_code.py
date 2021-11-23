import cv2
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import ndimage
import mplcursors
from numpy import linalg as LA

def get_points_selected():
    image1 = cv2.imread(f'./Q4/hallway1.jpg')
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    image2 = cv2.imread(f'./Q4/hallway2.jpg')
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    fig= plt.figure(figsize=(12, 4), constrained_layout=True)
    ax = [None for _ in range(2)]

    ax[0] = fig.add_subplot(1, 2, 1)
    a = ax[0].imshow(gray1, cmap='gray')

    ax[1] = fig.add_subplot(1, 2, 2)
    b = ax[1].imshow(gray2, cmap='gray')

    points_1 = []
    points_2 = []

    def on_click(sel):
        x = int(sel.target_[0])
        y = int(sel.target_[1])
        if sel.artist == a:
            points_1.append((y, x))
            sel.annotation.set_text(f'x: {x}\ny: {y} \npoint{len(points_1)}')
            # print(gray1[sel.target_[1]][sel.target_[0]])
        elif sel.artist == b:
            points_2.append((sel.target_[1], sel.target_[0]))
            sel.annotation.set_text(f'x: {x}, y: {y} \npoint{len(points_2)}')
    mplcursors.cursor(multiple=True).connect(
        "add", on_click)
    # print(cursor.artists)
    plt.show()

    assert len(points_1) == 4, 'You should select 4 points on figure 1'
    assert len(points_2) == 4, 'You should select 4 points on figure 2'
    return points_1, points_2

def matrix_for_point(point1, point2):
    '''Given a pair of point, produce a 2 x 9 matrix.
        x, y, 1, 0, 0, 0, -x'x, -x'y, -x'
        0, 0, 0, x, y, 1, -y'x, -y'y, -y
    '''
    res = np.array([[point1[0], point1[1], 1, 0, 0, 0, -point2[0] * point1[0], -point2[0] * point1[1], -point1[0]],
                    [0, 0, 0, point1[0], point1[1], 1, -point2[1] * point1[0], -point2[1] * point1[1], -point2[1]]])
    return res


def calculate_homography_matrix(points_1, points_2):
    A_matrix = np.array([[]]).reshape((0, 9))
    for i in range(4):
        point_matrix = matrix_for_point(points_1[i], points_2[i])
        A_matrix = np.vstack((A_matrix, point_matrix))

    m = np.matmul(A_matrix.T, A_matrix)
    w, v = LA.eig(m)
    smallest_index = np.argmin(w)
    h = v[ : , smallest_index].reshape((3, 3))
    return h

if __name__ == '__main__':
    points_1, points_2 = get_points_selected()
    print(points_1)
    print(points_2)
    calculate_homography_matrix(points_1, points_2)
