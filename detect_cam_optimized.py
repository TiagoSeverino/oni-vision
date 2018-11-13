from __future__ import division
import cv2
import numpy as np
import random
import math

#h, w = ( 172 // 16, 172 // 16 )
h, w = ( 172, 172 )
#h, w = ( int(172 * 1.5), int(172 * 1.5 ))

letters =	{
  "H": [cv2.imread('processed/h.png'), (255, 0, 0), 0.75 ],
  "S": [cv2.imread('processed/s.png'), (0, 255, 0), 0.6  ],
  "U": [cv2.imread('processed/u.png'), (0, 0, 255), 0.75 ]
}

for key, letter in letters.items():
	template = letter[0]

	template = cv2.resize(template, (w, h), interpolation=cv2.INTER_NEAREST)
	template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
	ret, template = cv2.threshold(template, 10, 255, cv2.THRESH_BINARY)
	letters[key][0] = template


def Match(img, template, threshold):
	res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
	loc = np.where( res >= threshold)

	#if np.amax(res) >= threshold:
	#	certainty = int(np.amax(res)*100)
	#	print("Certainty: {}%".format(certainty))

	if zip(*loc[::-1]):
		return True
	else:
		return False

def RemoveNoise(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	ret, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV) #  + cv2.THRESH_OTSU

	kernel = np.ones((6, 6),'uint8')

	binary = cv2.erode(binary,kernel,iterations=1)
	binary = cv2.dilate(binary,kernel,iterations=1)

	return binary

def getExtremePoints(c, img = None):
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])

	if (img is not None):
		cv2.drawContours(img, [c], -1, (0, 255, 255), -1)
		cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
		cv2.circle(img, extRight, 8, (0, 255, 0), -1)
		cv2.circle(img, extTop, 8, (255, 0, 0), -1)
		cv2.circle(img, extBot, 8, (255, 255, 0), -1)

	return extLeft[0], extRight[0], extTop[1], extBot[1]

def getAngle(img):
	coords = np.column_stack(np.where(img > 0))
	angle = cv2.minAreaRect(coords)[-1]

	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle

	return angle

def rotateAndCrop(angle, img, w, h):
	(_h, _w) = img.shape[:2]
	center = (_w // 2, _h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	img = cv2.warpAffine(img, M, (_w, _h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

	_, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	if contours:
		Left, Right, Top, Bot = getExtremePoints(contours[0])

		witdh = Right - Left
		height = Bot - Top

		if witdh != 0 and height != 0:
			img = img[ Top:(Top + height), Left:(Left + witdh)]
			img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
	return img


cap = cv2.VideoCapture('test/1.webm')

if cap.isOpened():
    print("Device Opened\n")
else:
    print("Failed to open Device\n")
    exit(1)

e0 = cv2.getTickCount()

while(cap.isOpened()):

	ret, img = cap.read()

	if not ret:
		break

	#img = cv2.resize(img, (640, 480))

	binary = RemoveNoise(img)
	#cv2.imshow("Binary", binary)

	_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	for c in contours:
		Left, Right, Top, Bot =  getExtremePoints(c, img = None)

		witdh = Right- Left
		height = Bot - Top

		if witdh == 0 or height == 0: # or witdh < 80 or height < 80:
			continue

		cropped = binary[ Top:(Top + height), Left:(Left + witdh)]
		cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)

		# H / U

		l_cropped = cropped[:, :w//2]
		r_cropped = cropped[:, w//2:w]

		top_left = [0, 0]
		top_right = [0, 0]
		
		_, contours, hierarchy = cv2.findContours(l_cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			top_left = tuple(contours[0][contours[0][:, :, 1].argmin()][0])

		_, contours, hierarchy = cv2.findContours(r_cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			top_right = tuple(contours[0][contours[0][:, :, 1].argmin()][0])
			top_right = (top_right[0] + w//2, top_right[1])

		#cv2.line(cropped, top_left, top_right,(255, 255, 0),5)

		_width = top_right[0] - top_left[0]
		_height = abs(top_right[1] - top_left[1])

		angle = math.degrees(math.atan(_height / _width))
		if top_right[1] < top_left[1]:
			angle = -angle

		cropped1 = rotateAndCrop(angle, cropped.copy(), w, h)

		# S
		angle2 = getAngle(cropped)

		cropped2 = rotateAndCrop(angle2, cropped.copy(), w, h)

		for key, letter in letters.items():
			if Match(cropped1 if key != "S" else cropped2, template = letter[0], threshold = letter[2]):
				cv2.drawContours(img, [c], -1, letter[1], -1)
				#cv2.rectangle(img, (Left, Top), (Right, Bot), (0,0,0), -1)	
				cv2.imshow("Cropped: " + key, cropped1 if key != "S" else cropped2)
				#cv2.imwrite('matched/' + key + "/" + str(random.random()) + ".png", cropped)
				continue

	cv2.imshow("Frame", img)
	#cv2.imwrite('false/' + str(random.random()) + ".png", binary)

	if cv2.waitKey(90) & 0xFF == ord('q'):
		break

e2 = cv2.getTickCount()
t = (e2 - e0)/cv2.getTickFrequency()
print( "Processing time: {}".format(t))

cap.release()
cv2.destroyAllWindows()