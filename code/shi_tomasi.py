import cv2
import numpy as np

img = cv2.imread('../images/image2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int8(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 4, (0, 255, 0), -1)

cv2.imwrite('../results/shi_tomasi_result.jpg', img)
