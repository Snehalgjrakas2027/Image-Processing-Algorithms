import cv2
import sys

# Load grayscale images
img1 = cv2.imread('../images/image1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../images/image2.png', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if img1 is None:
    print("Error: Cannot load image1.png")
    sys.exit(1)

if img2 is None:
    print("Error: Cannot load image2.png")
    sys.exit(1)

# ORB as alternative to SURF
orb = cv2.ORB_create(nfeatures=1000)

# Detect and compute keypoints
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

# Save output
cv2.imwrite('../results/orb_result.jpg', result_img)
