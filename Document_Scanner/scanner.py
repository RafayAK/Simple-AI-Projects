# import the necessary packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

# read image, clone it, calculate ratio between original image height
# and new image height and resize to target height
image = cv2.imread(args["image"])
orig = image.copy()
ratio = image.shape[0]/500.0
image = imutils.resize(image=image, height=500)

# convert to gray scale, blur it and do edge detection
gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(src=gray, ksize=(5,5), sigmaX=0)
edged = cv2.Canny(image=image,threshold1=75, threshold2=200)

# show og image and edged image
print("Creating edged image...")
cv2.imshow("image", image)
cv2.imshow("edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find contours
contours = cv2.findContours(image=edged.copy(), mode = cv2.RETR_LIST,
                            method=cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# loop over contours
for c in contours:
    # approximate contours
    perimeter = cv2.arcLength(curve=c, closed=True)
    approx = cv2.approxPolyDP(curve=c, epsilon=0.02*perimeter, closed=True)

    if len(approx) == 4:
        # found the doc in the image
        screenContour= approx
        break

# show the outline(contour) of the doc
print('Calculating outline of the document...')

cv2.drawContours(image=image, contours=[screenContour], contourIdx=-1,
                 color=(0,255,0), thickness=2)
cv2.imshow('outline', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply a perspective transform on the original image
warped = four_point_transform(orig, screenContour.reshape(4,2)*ratio)

# convert warped img to gray scale, perform thresholding
# to give it that "scanned" look
warped = cv2.cvtColor(src=warped, code=cv2.COLOR_BGR2GRAY)
thresh_image = threshold_local(image=warped, block_size=81,
                               offset=10, method="gaussian")
warped = (warped > thresh_image).astype("uint8")*255

# print the orignal and the scanned image
print("Generating scanned image...")
cv2.imshow('original', imutils.resize(orig, height=650))
cv2.imshow('scanned', imutils.resize(warped, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()


