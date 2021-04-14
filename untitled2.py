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



def triangle_area(tri):
    x1, y1, x2, y2, x3, y3 = tri[0][0], tri[0][1], tri[1][0], tri[1][1], tri[2][0], tri[2][1]
    return abs(0.5 * (((x2-x1)*(y3-y1))-((x3-x1)*(y2-y1))))

def leftEar(shape):
	p111 = shape[36:37][0]
	p112 = shape[37:38][0]
	p113 = shape[41:42][0]

	tri11 = np.array([p111, p112, p113])

	p121 = shape[21:22][0]
	p122 = shape[20:21][0]
	p123 = shape[38:39][0]

	tri12 = np.array([p121, p122, p123])

	p131 = shape[38:39][0]
	p132 = shape[39:40][0]
	p133 = shape[40:41][0]

	tri13 = np.array([p131, p132, p133])

	p141 = shape[17:18][0]
	p142 = shape[18:19][0]
	p143 = shape[37:38][0]

	tri14 = np.array([p141, p142, p143])

	ear = (triangle_area(tri11) + triangle_area(tri13)) / (triangle_area(tri12) + triangle_area(tri14))

	leftEyeHull = cv2.convexHull(tri11)
	rightEyeHull = cv2.convexHull(tri12)

	tri3hull = cv2.convexHull(tri13)
	tri4hull = cv2.convexHull(tri14)


	cv2.drawContours(frame, [tri11], -1, (255, 255, 0), 1)
	cv2.drawContours(frame, [tri12], -1, (255, 255, 0), 1)

	cv2.drawContours(frame, [tri13], -1, (255, 255, 0), 1)
	cv2.drawContours(frame, [tri14], -1, (255, 255, 0), 1)
	return ear

def rightEar(shape):
	p111 = shape[42:43][0]
	p112 = shape[43:44][0]
	p113 = shape[47:48][0]

	tri11 = np.array([p111, p112, p113])

	p121 = shape[26:27][0]
	p122 = shape[25:26][0]
	p123 = shape[44:45][0]

	tri12 = np.array([p121, p122, p123])

	p131 = shape[44:45][0]
	p132 = shape[45:46][0]
	p133 = shape[46:47][0]

	tri13 = np.array([p131, p132, p133])

	p141 = shape[22:23][0]
	p142 = shape[23:24][0]
	p143 = shape[43:44][0]

	tri14 = np.array([p141, p142, p143])

	ear = (triangle_area(tri11) + triangle_area(tri13)) / (triangle_area(tri12) + triangle_area(tri14))

	leftEyeHull = cv2.convexHull(tri11)
	rightEyeHull = cv2.convexHull(tri12)

	tri3hull = cv2.convexHull(tri13)
	tri4hull = cv2.convexHull(tri14)


	cv2.drawContours(frame, [tri11], -1, (255, 255, 0), 1)
	cv2.drawContours(frame, [tri12], -1, (255, 255, 0), 1)

	cv2.drawContours(frame, [tri13], -1, (255, 255, 0), 1)
	cv2.drawContours(frame, [tri14], -1, (255, 255, 0), 1)
	return ear

def doubleClick():
	mPoint = pyautogui.position();
	pyautogui.doubleClick(mPoint.x,mPoint.y)
	return

def leftClick():
	mPoint = pyautogui.position();
	pyautogui.leftClick(mPoint.x,mPoint.y)
	return

def rightClick():
	mPoint = pyautogui.position();
	pyautogui.rightClick(mPoint.x,mPoint.y)
	return

def moveMouse(x, y):
	pyautogui.moveRel(x, y)
	return

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="blink_detection_demo.mp4",
	help="path to input video file")
args = vars(ap.parse_args())


EYE_AR_THRESH = 0.13
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0
FLAG = 0
TIME_COUNTER = False
nose_tip_position = {
		'x' : 0,
		'y' : 0
	}

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]


print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
fileStream = True

time.sleep(1.0)

cap = cv2.VideoCapture("https://192.168.43.1:8080/video")
point1 = []
point2 = []
point3 = []

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

		ear = leftEar(shape)
		ear = np.tanh(ear)
		rear = rightEar(shape)
		rear = np.tanh(rear)
#(ear1+ear2)/2

# 		ear = np.tanh(ear)
		#print(type(tri1))


# 		p211 =
# 		p212 =
# 		p213 =
#
# 		tri3 = [p211, p212, p213]
#
# 		p221 =
# 		p222 =
# 		p223 =
#
# 		tri4 = [p221, p222, p223]
		if ear > EYE_AR_THRESH:
			ear = 0.22
		if rear > EYE_AR_THRESH:
			rear = 0.22
		point1.append(ear)
		point2.append(rear)
		point3.append((ear + rear) / 2)
		plt.plot(point1)
		plt.plot(point2)
		plt.plot(point3)




		if ear < EYE_AR_THRESH:
 			COUNTER += 1
 			if FLAG == 0:
				 FLAG +=1
		else:
			if FLAG == 1:
				 FLAG +=1

		if FLAG == 2:
 			TOTAL +=1
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
				 print("double Click")
				 doubleClick()
 			elif TOTAL == 4:
				 print("left click")
				 leftClick()
 			elif TOTAL == 3:
				 print("right click")
				 rightClick()
 			TOTAL = 0



		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(rear), (300, 50),
 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.circle(frame, tuple(nose),2,(0,0,255),2)

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