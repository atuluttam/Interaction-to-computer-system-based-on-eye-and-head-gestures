from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyautogui
import matplotlib.pyplot as plt

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def doubleClick():
    mPoint = pyautogui.position()
    pyautogui.doubleClick(mPoint.x, mPoint.y)
    return

def leftClick():
    mPoint = pyautogui.position()
    pyautogui.leftClick(mPoint.x, mPoint.y)
    return

def rightClick():
    mPoint = pyautogui.position()
    pyautogui.rightClick(mPoint.x, mPoint.y)
    return


def moveMouse(x, y):
    pyautogui.moveRel(x, y)
    return


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="blink_detection_demo.mp4",
                help="path to input video file")
args = vars(ap.parse_args())


EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0
FLAG = 0
TIME_COUNTER = False
nose_tip_position = {
    'x': 0,
    'y': 0
}

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

print(face_utils.FACIAL_LANDMARKS_68_IDXS)
print(face_utils.FACIAL_LANDMARKS_5_IDXS)
print(face_utils.FACIAL_LANDMARKS_IDXS)

print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
fileStream = True

time.sleep(1.0)
point = []
point2 = []
cap = cv2.VideoCapture("https://192.168.43.1:8080/video")


def func(frame):
    global FLAG
    global TOTAL
    global TIME_COUNTER
    global COUNTER
    global start_time
    global nose_tip_position

# 	frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        nose = shape[30:31][0]
        nose_tip_position['x'] = nose[0]
        nose_tip_position['y'] = nose[1]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
# 		print(shape[])
        [x, y, w, h] = cv2.boundingRect(shape[36:42])
        ri = frame[y:y+h, x:x+w]
        ri = imutils.resize(ri, width=250, inter=cv2.INTER_CUBIC)
        ear = (leftEAR + rightEAR) / 2.0

        point.append(ear)
        plt.plot(point)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if FLAG == 0:
                FLAG += 1
        else:
            if FLAG == 1:
                FLAG += 1

        if FLAG == 2:
            TOTAL += 1
            FLAG = 0
            COUNTER = 0

#
        if TOTAL > 0 and TIME_COUNTER == False:
            start_time = time.time()
            TIME_COUNTER = True

        elapsed_time = 0
        if TIME_COUNTER == True:
            elapsed_time = time.time() - start_time

        if elapsed_time > 2.5 and TIME_COUNTER == True:
            TIME_COUNTER = False
            elapsed_time = 0
            if TOTAL == 2:
                doubleClick()
                print("double Click")
            elif TOTAL == 4:
                leftClick()
                print("left click")
            elif TOTAL == 3:
                rightClick()
                print("right click")
            TOTAL = 0

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "LEAR: {:.2f}".format(leftEAR), (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "REAR: {:.2f}".format(rightEAR), (300, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.circle(frame, tuple(nose), 2, (0, 0, 255), 2)

        cv2.imshow("roi", ri)
    cv2.imshow("Frame", frame)


while True:
    ret, frame = cap.read()
    if ret is False:
        break
    func(frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

plt.show()
# do a bit of cleanup
cv2.destroyAllWindows()
