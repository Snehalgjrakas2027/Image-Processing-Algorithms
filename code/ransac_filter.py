import cv2
import numpy as np

img1 = cv2.imread('../images/image1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../images/image2.png', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
pts1 = []
pts2 = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

# Apply RANSAC
M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()

result_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, matchColor=(0,255,0),
                             matchesMask=matches_mask, flags=2)
cv2.imwrite('../results/ransac_result.jpg', result_img)
