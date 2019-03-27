from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import argparse


def midpoint(ptA, ptB):
    return ((ptA[0]+ptB[0]) * 0.5 , (ptA[1]+ptB[1]) *0.5)

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')

ap.add_argument('-w', '--width', required=True, type=float,
                help='width of the left most object int the image, the reference object(in inches)')

args = vars(ap.parse_args())

# load the image, convert it to grays cale, and blur it slightly
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7,7), 0)

# perform edge detection, then perform dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, kernel=None, iterations=1)
edged = cv2.erode(edged, kernel= None, iterations=1)

# find the contours in the edged map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# loop over the contours individually
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 100:
        continue

    # compute rotated bounding box of the contour
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in the top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # loop over the corners and draw them
    for (x, y) in box:
        cv2.circle(orig, center=(int(x), int(y)), radius=5,
                   color=(0, 0, 255), thickness=-1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and the top-right coordinates, followed
    # by the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and bottom-left points,
    # followed by the midpoint between the top-right and bottom-right
    # points
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(orig, pt1=(int(tltrX), int(tltrY)), pt2=(int(blbrX), int(blbrY)),
             color=(255, 0, 255), thickness=2)
    cv2.line(orig, pt1=(int(tlblX), int(tlblY)), pt2=(int(trbrX), int(trbrY)),
             color=(255, 0, 255), thickness=2)

    # compute Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # if the pixels per metric has not been initialize, then
    # compute it as the ratio of pixels to the supplied metric
    # (the size of coin in this case, in inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args['width']

    # now we can compute the size of object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f}in".format(dimA), org=(int(tltrX - 15), int(tltrY - 10)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(255, 255, 255), thickness=2)
    cv2.putText(orig, "{:.1f}in".format(dimB), org=(int(trbrX + 10), int(trbrY)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(255, 255, 255), thickness=2)

    # plot the images
    cv2.imshow('image', orig)
    cv2.waitKey(0)

# do some clean up
cv2.destroyAllWindows()