# std
from random import randint

# 3p
import numpy as np
import cv2

# Sandbox for Tesson
image = cv2.imread('./samples/first.jpg')

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

dest = None
ret, tresh_img = cv2.threshold(image, 205, 255, cv2.THRESH_TOZERO)

# Conversion to a binary image
ret, tresh_bin = cv2.threshold(cv2.cvtColor(tresh_img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

cv2.namedWindow("Tresh", cv2.WINDOW_NORMAL)

# For each column compute the histogram
def pixel_intensity(pixel):
    return pixel[0]/3 + pixel[1]/3 + pixel[2]/3


i = 0
peak = []
while i < tresh_img.shape[1]:
    col = tresh_img[:,i]
    sm = 0
    coef = 0
    for j in range(0, len(col)-1):
        intensity = pixel_intensity(col[j])
        coef += intensity
        sm += intensity * j
    if coef == 0:
        peak.append(-1)
    else:
        peak.append(sm/coef)
    i += 1

# Add the cloud on top of the original image
for j, p in enumerate(peak):
    if p is not None:
        cv2.circle(tresh_img, (j, p), 2, (255, 0, 0))

# Compute the difference (in meters) between the base line and the top

# TODO: make this parametrizable
ALPHA = 1

## Let's determine the base line as the line going through two points close to
## the picture's boudaries
p1 = (10, peak[10])
p2 = (len(peak)-11, peak(len(peak)-11))


# contourds, hierachy = cv2.findContours(tresh_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# i = 0
# final_img = np.zeros((image.shape[0], image.shape[1], image.shape[2]), np.uint8)
# for c in contours:
#     red_amnt = 120 + randint(0, 135)
#     green_amnt = 120 + randint(0, 135)
#     blue_amnt = 120 + randint(0, 135)
#
#     if len(c) < 20:
#         continue
#
#     i += 1
#     print(i)
#     for l in c:
#         # cv2.circle(tresh_img, (int(l[0][0]), int(l[0][1])), 2, (red_amnt, green_amnt, blue_amnt))
#         cv2.circle(final_img, (int(l[0][0]), int(l[0][1])), 2, (red_amnt, green_amnt, blue_amnt))
#
cv2.imshow('Tresh', tresh_img)
# cv2.imshow('Test',image)
#
# cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
# cv2.imshow('Contours', final_img)
#
cv2.waitKey(0)
cv2.destroyAllWindows()
