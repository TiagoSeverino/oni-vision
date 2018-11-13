import cv2
import numpy as np
import random

letter = 'U.png'

template = cv2.imread('processed/' + letter)
h, w, channel = template.shape
template = cv2.resize(template, (w // 8, h // 8))
template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
ret, template = cv2.threshold(template, 10, 255, cv2.THRESH_BINARY)

h, w = template.shape

img = cv2.imread('test/1.jpg')
img = cv2.resize(img, (960, 540))

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)

kernel = np.ones((6, 6),'uint8')

binary = cv2.erode(binary,kernel,iterations=1)
binary = cv2.dilate(binary,kernel,iterations=1)
cv2.imshow("Binary", binary)

_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])

	cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
	cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
	cv2.circle(img, extRight, 8, (0, 255, 0), -1)
	cv2.circle(img, extTop, 8, (255, 0, 0), -1)
	cv2.circle(img, extBot, 8, (255, 255, 0), -1)

	witdh = extRight[0] - extLeft[0]
	height = extBot[1] - extTop[1]

	if witdh == 0 or height == 0:
		continue

	cropped = binary[ extTop[1]:(extTop[1] + height), extLeft[0]:(extLeft[0] + witdh)]

	cropped = cv2.resize(cropped, (w, h))

	coords = np.column_stack(np.where(cropped > 0))
	angle = cv2.minAreaRect(coords)[-1]

	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle

	if angle != 0:
		print("Angle: {}".format(angle))

		(h1, w1) = cropped.shape[:2]
		center = (w1 // 2, h1 // 2)
		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		cropped = cv2.warpAffine(cropped, M, (w1, h1), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

	res = cv2.matchTemplate(cropped, template, cv2.TM_CCOEFF_NORMED)
	threshold = 0.6
	loc = np.where( res >= threshold)

	if zip(*loc[::-1]):
		cv2.rectangle(img, (extLeft[0], extTop[1]), (extRight[0], extBot[1]), (0,0,255), 2)
		cv2.imshow("Cropped: " + str(random.random()), cropped)

cv2.imshow("Contours", img)

cv2.waitKey(0)
cv2.destroyAllWindows()