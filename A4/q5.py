import cv2
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from numpy import linalg as LA

# cap = cv2.VideoCapture(0) # camera feed
cap = cv2.VideoCapture('./Q4/q5.mp4')
# capture one frame
ret,frame = cap.read()

# detect a face on the first frame
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
face_boxes = face_detector.detectMultiScale(frame) 

if len(face_boxes)==0:
    print('no face detected')
    assert(False)

# initialize the tracing window around the (first) detected face
(x,y,w,h) = tuple(face_boxes[0]) 
track_window = (x,y,w,h)
#  region of interest for tracking
roi = frame[y:y+h, x:x+w]

# convert the roi to HSV so we can construct a histogram of Hue 
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# why do we need this mask? (remember the cone?)
# read the description for Figure 3 in the original Cam Shift paper: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.7673 
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

# form histogram of hue in the roi
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])

# normalize the histogram array values so they are in the min=0 to max=255 range
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# termination criteria for mean shift: 10 iteration or shift less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

def iou_calc(A, B):
	# rectangle A and B are defined as (x, y , w, h)
    A = [A[0], A[1], A[0] + A[2], A[1] + A[3]]
    B = [B[0], B[1], B[0] + B[2], B[1] + B[3]]
    x_upper_left = max(A[0], B[0])
    y_upper_left = max(A[1], B[1])
    x_lower_right = min(A[2], B[2])
    y_lower_right = min(A[3], B[3])
	# compute the area of intersection rectangle
    intersection_area = max(0, x_lower_right - x_upper_left + 1) * max(0, y_lower_right - y_upper_left + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxA_area = (A[2] - A[0] + 1) * (A[3] - A[1] + 1)
    boxB_area = (B[2] - B[0] + 1) * (B[3] - B[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    # return the intersection over union value
    return iou
# Frame starts at 2
i = 2
LOWER_THRESHOLD = 0.62
HIGHER_THRESHOLD = 0.68
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
    ret ,frame = cap.read() 
    
    if ret == True: 
  
        # convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # histogram back projection using roi_hist 
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # use meanshift to shift the tracking window
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        # display tracked window
        x,y,w,h = track_window
        img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),5)

        # detect a face on the first frame
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
        face_boxes = face_detector.detectMultiScale(frame) 

        if len(face_boxes)==0:
            print('no face detected')
            assert(False)
        
        iou_all_faces = []
        for j in range(len(face_boxes)):
            iou = iou_calc((x, y, w, h), tuple(face_boxes[j]))
            iou_all_faces.append(iou)
        iou = max(iou_all_faces)
        iou_list.append(iou)
        x,y,w,h = tuple(face_boxes[np.argmax(iou_all_faces)])
        img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),5)
        if iou < lowest_iou_value:
            lowest_iou_value = iou
            lowest_match_frame = img
        
        if iou > highest_iou_value:
            highest_iou_value = iou
            highest_match_frame = img
        
        if iou < LOWER_THRESHOLD:
            num_below += 1
        
        if iou > HIGHER_THRESHOLD:
            num_above += 1
        frames.append(i)
        i += 1
        cv2.imshow('mean shift tracking demo',img)
        if cv2.waitKey(33) & 0xFF == 27: # wait a bit and exit is ESC is pressed
            break
        
    else:
        break

cv2.destroyAllWindows()
cap.release()
plt.plot(frames, iou_list)
plt.show()
cv2.imshow("Lowest matching frame" ,lowest_match_frame)
cv2.imshow("Hightest matchin frame",highest_match_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f'{num_above / len(frames) * 100 :.2f}% of the frames in which iou is larger then {HIGHER_THRESHOLD}')
print(f'{num_below / len(frames) * 100 :.2f}% of the frames in which iou is smaller then {LOWER_THRESHOLD}')