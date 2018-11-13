import cv2
import numpy as np
import random

#h, w = ( 172 // 8, 172 // 8 )
h, w = ( 172, 172 )
#h, w = ( int(172 * 1.5), int(172 * 1.5 ))

h_template = cv2.imread('processed/h.png')
h_template = cv2.resize(h_template, (w, h))
h_template = cv2.cvtColor(h_template,cv2.COLOR_BGR2GRAY)
ret, h_template = cv2.threshold(h_template, 10, 255, cv2.THRESH_BINARY)

s_template = cv2.imread('processed/s.png')
s_template = cv2.resize(s_template, (w, h))
s_template = cv2.cvtColor(s_template,cv2.COLOR_BGR2GRAY)
ret, s_template = cv2.threshold(s_template, 10, 255, cv2.THRESH_BINARY)

u_template = cv2.imread('processed/u.png')
u_template = cv2.resize(u_template, (w, h))
u_template = cv2.cvtColor(u_template,cv2.COLOR_BGR2GRAY)
ret, u_template = cv2.threshold(u_template, 10, 255, cv2.THRESH_BINARY)

cap = cv2.VideoCapture('test/1.webm')

if cap.isOpened():
    print("Device Opened\n")
else:
    print("Failed to open Device\n")
    exit(1)

e0 = cv2.getTickCount()

while(cap.isOpened()):

	#img = cv2.imread('test/1.jpg')
	ret, img = cap.read()

	if not ret:
		break

	# img = cv2.resize(img, (960, 540)) # (960, 540)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	# binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)
	ret, binary = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	kernel = np.ones((6, 6),'uint8')

	binary = cv2.erode(binary,kernel,iterations=1)
	binary = cv2.dilate(binary,kernel,iterations=1)

	#cv2.imshow("Binary", binary)

	_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	for c in contours:
		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extTop = tuple(c[c[:, :, 1].argmin()][0])
		extBot = tuple(c[c[:, :, 1].argmax()][0])

		#cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
		#cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
		#cv2.circle(img, extRight, 8, (0, 255, 0), -1)
		#cv2.circle(img, extTop, 8, (255, 0, 0), -1)
		#cv2.circle(img, extBot, 8, (255, 255, 0), -1)

		witdh = extRight[0] - extLeft[0]
		height = extBot[1] - extTop[1]

		if witdh < 20 or height < 20:
			continue

		cropped = binary[ extTop[1]:(extTop[1] + height), extLeft[0]:(extLeft[0] + witdh)]

		cropped = cv2.resize(cropped, (w, h))

		coords = np.column_stack(np.where(cropped > 0))
		angle = cv2.minAreaRect(coords)[-1]

		if angle < -45:
			angle = -(90 + angle)
		else:
			angle = -angle

		if angle < -0 or angle > 0:
			#print("Angle: {}".format(angle))

			(h1, w1) = cropped.shape[:2]
			center = (w1 // 2, h1 // 2)
			M = cv2.getRotationMatrix2D(center, angle, 1.0)
			cropped = cv2.warpAffine(cropped, M, (w1, h1), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

			_, contours, hierarchy = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			if contours:
				cc = contours[0]
				cextLeft = tuple(cc[cc[:, :, 0].argmin()][0])
				cextRight = tuple(cc[cc[:, :, 0].argmax()][0])
				cextTop = tuple(cc[cc[:, :, 1].argmin()][0])
				cextBot = tuple(cc[cc[:, :, 1].argmax()][0])

				witdh = cextRight[0] - cextLeft[0]
				height = cextBot[1] - cextTop[1]

				if witdh != 0 and height != 0:
					cropped = cropped[ cextTop[1]:(cextTop[1] + height), cextLeft[0]:(cextLeft[0] + witdh)]
					cropped = cv2.resize(cropped, (w, h))

		res = cv2.matchTemplate(cropped, h_template, cv2.TM_CCORR_NORMED)
		threshold = 0.725
		loc = np.where( res >= threshold)

		if zip(*loc[::-1]):
			cv2.drawContours(img, [c], -1, (0, 0, 255), -1)
			#cv2.rectangle(img, (extLeft[0], extTop[1]), (extRight[0], extBot[1]), (0,0,255), -1)
			#cv2.imshow("Cropped", cropped)

		res = cv2.matchTemplate(cropped, s_template, cv2.TM_CCORR_NORMED)
		threshold = 0.66
		loc = np.where( res >= threshold)

		if zip(*loc[::-1]):
			cv2.drawContours(img, [c], -1, (0, 255, 0), -1)
			#cv2.rectangle(img, (extLeft[0], extTop[1]), (extRight[0], extBot[1]), (0,255,0), -1)
			#cv2.imshow("Cropped", cropped)

		res = cv2.matchTemplate(cropped, u_template, cv2.TM_CCORR_NORMED)
		threshold = 0.73
		loc = np.where( res >= threshold)

		if zip(*loc[::-1]):
			cv2.drawContours(img, [c], -1, (255, 0, 0), -1)
			#cv2.rectangle(img, (extLeft[0], extTop[1]), (extRight[0], extBot[1]), (255,0,0), -1)
			#cv2.imshow("Cropped", cropped)

	cv2.imshow("Contours", img)

	if cv2.waitKey(30) & 0xFF == ord('q'):
		break

e2 = cv2.getTickCount()
t = (e2 - e0)/cv2.getTickFrequency()
print( "Processing time: {}".format(t))

cap.release()
cv2.destroyAllWindows()