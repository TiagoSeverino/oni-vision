import cv2
import numpy as np

letter = 'U.png'

img = cv2.imread('original/' + letter)

cv2.imshow("Original", img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)

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

	print("Ext Left {}, Ext Right {}, Ext Top {}, Ext Bot {}".format(extLeft, extRight, extTop, extBot))

	witdh = extRight[0] - extLeft[0]
	height = extBot[1] - extTop[1]
	print("Witdh: {}, Height: {}".format(witdh, height))

	cropped = binary[ extTop[1]:(extTop[1] + height), extLeft[0]:(extLeft[0] + witdh)]

	cropped = cv2.resize(cropped, (172, 172 )) # (128, 128 * height / witdh )
	cv2.imshow("Cropped", cropped)

	cv2.imwrite('processed/' + letter, cropped)

cv2.imshow("Contours",img)


cv2.waitKey(0)
cv2.destroyAllWindows()