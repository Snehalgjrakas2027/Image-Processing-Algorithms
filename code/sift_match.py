import cv2
import matplotlib.pyplot as plt

# Load PNG images in grayscale
img1 = cv2.imread('../images/image1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../images/image2.png', cv2.IMREAD_GRAYSCALE)

# Check if images were loaded properly
if img1 is None or img2 is None:
    print("Error: One or both images not found. Make sure they are named correctly and located in the 'images' folder.")
    exit()

# SIFT detector
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Feature matching with BFMatcher + Ratio Test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Draw matches
result_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)

# Save result
cv2.imwrite('../results/sift_result.png', result_img)
print("SIFT matching result saved as sift_result.png in 'results' folder.")
