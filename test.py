import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import angle
import cv2
# cap = cv.VideoCapture(0)
# ret, frame1 = cap.read()
# prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# hsv[..., 1] = 255
# while(1):
#     ret, frame2 = cap.read()
#     next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
#     flow = cv.calcOpticalFlowFarneback(
#         prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv[..., 0] = ang*180/np.pi/2
#     hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
#     bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#     cv.imshow('frame2', bgr)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv.imwrite('opticalfb.png', frame2)
#         cv.imwrite('opticalhsv.png', bgr)
#     prvs = next

# image = cv2.imread('./A3/Q3/1.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(gray.shape)
# print(np.cos(np.pi / 2))
# x_pos = [10, 10]
# y_pos = [10, 10]
# x_direct = [np.cos(np.pi / 6) * 30, np.cos(np.pi / 2) * 10]
# y_direct = [np.sin(np.pi / 6) * 30, np.sin(np.pi / 2) * 10]

# # plt.imshow(gray, cmap='gray')
# plt.quiver(x_pos, y_pos, x_direct, y_direct, 1,
#            headlength=0, headwidth=1, pivot='mid')

# plt.show()


def retriving_mag_and_angle(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    # Calculate the magnitude and angle for sourc.e image
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # Convert direct angle to indirect angle

    # def convert_to_indirect_angle(x):
    #     if x >= 165 and x < 345:
    #         return x - 180
    #     elif x >= 345:
    #         return x - 360
    #     else:
    #         return x
    # convert_to_indirect_angle = np.vectorize(convert_to_indirect_angle)
    # angle = convert_to_indirect_angle(angle)
    # Threshold magnitude
    t = 0
    threshold = np.vectorize(lambda x: x if x >= t else 0)
    mag = threshold(mag)
    # plt.imshow(mag, cmap='gray')
    # plt.show()
    return mag, angle


a = np.array([1, 2])
b = np.array([3, 4])
print(a * b)
