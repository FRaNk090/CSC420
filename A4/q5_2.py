import cv2
import numpy as np
import matplotlib.pyplot as plt

# cap = cv2.VideoCapture(0) # camera feed
cap = cv2.VideoCapture('./Q4/q5.mp4')
# capture one frame
ret, frame = cap.read()

# detect a face on the first frame
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_boxes = face_detector.detectMultiScale(frame)

if len(face_boxes) == 0:
    print('no face detected')
    assert(False)

# initialize the tracing window around the (first) detected face
(x, y, w, h) = tuple(face_boxes[0])
track_window = (x, y, w, h)
#  region of interest for tracking
roi = frame[y:y+h, x:x+w]

# convert the roi to gray so we can construct a histogram of gradient angle of gray
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Calculate derivatives
gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1)
# Calculate the magnitude and angle for source image
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

max_mag = np.max(mag)

# Define the mask based on its gradient magnitude
mask = cv2.inRange(mag, 0.05 * max_mag, float(max_mag))

# form histogram of hue in the roi with 24 bins
roi_hist = cv2.calcHist([angle], [0], mask, [24], [0, 360])

# normalize the histogram array values so they are in the min=0 to max=255 range
cv2.normalize(roi_hist, roi_hist, 0, 360, cv2.NORM_MINMAX)

# termination criteria for mean shift: 10 iteration or shift less than 1 pixel
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


def iou_calc(A, B):
    '''Given two squares in form of (x, y, w, h), 
       it will return the iou value
    '''
    # rectangle A and B are defined as (x, y , w, h)
    A = [A[0], A[1], A[0] + A[2], A[1] + A[3]]
    B = [B[0], B[1], B[0] + B[2], B[1] + B[3]]
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    AArea = (A[2] - A[0] + 1) * (A[3] - A[1] + 1)
    BArea = (B[2] - B[0] + 1) * (B[3] - B[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(AArea + BArea - interArea)
    # return the intersection over union value
    return iou


# Frame starts at 2
i = 2
frames = []
iou_list = []
LOWER_THRESHOLD = 0.1
HIGHER_THRESHOLD = 0.5
frames = []
iou_list = []
lowest_match_frame = None
lowest_iou_value = np.Inf
highest_match_frame = None
highest_iou_value = -np.Inf
num_below = 0
num_above = 0

while True:

    # grab a frame
    ret, frame = cap.read()
    if ret == True:
        # detect a face on the first frame
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_boxes = face_detector.detectMultiScale(frame)

        # convert to HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        # Calculate the magnitude and angle for source image
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # histogram back projection using roi_hist
        dst = cv2.calcBackProject([angle], [0], roi_hist, [0, 360], 1)

        # use meanshift to shift the tracking window
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # display tracked window
        x, y, w, h = track_window
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)

        if len(face_boxes) == 0:
            frames.append(i)
            i += 1
            iou_list.append(0.)
            print('no face detected')
            continue

        # Store largest iou value
        iou_all_faces = []
        for j in range(len(face_boxes)):
            iou = iou_calc((x, y, w, h), tuple(face_boxes[j]))
            iou_all_faces.append(iou)
        iou = max(iou_all_faces)
        iou_list.append(iou)
        x, y, w, h = tuple(face_boxes[np.argmax(iou_all_faces)])
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

        # Record the lowest match frame and highest match frame
        if iou < lowest_iou_value:
            lowest_iou_value = iou
            lowest_match_frame = img

        if iou > highest_iou_value:
            highest_iou_value = iou
            highest_match_frame = img

        # Record the number of frames that are above threshold and below the threshold
        if iou < LOWER_THRESHOLD:
            num_below += 1

        if iou > HIGHER_THRESHOLD:
            num_above += 1
        frames.append(i)
        i += 1
        cv2.imshow('mean shift tracking demo', img)
        if cv2.waitKey(33) & 0xFF == 27:  # wait a bit and exit is ESC is pressed
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()
plt.plot(frames, iou_list)
plt.show()
cv2.imshow("Lowest matching frame", lowest_match_frame)
cv2.imshow("Hightest matchin frame", highest_match_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f'{num_above / len(frames) * 100 :.2f}% of the frames in which iou is larger then {HIGHER_THRESHOLD}')
print(f'{num_below / len(frames) * 100 :.2f}% of the frames in which iou is smaller then {LOWER_THRESHOLD}')
