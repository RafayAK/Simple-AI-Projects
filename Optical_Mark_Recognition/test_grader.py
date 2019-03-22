# import necessary pakages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
import argparse

# construct argparser to pass in the argument for image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                  help="path to the input image")
args = vars(ap.parse_args())

# define the answer key which maps question->answer
ANSWER_KEY = {0:1, 1:4, 2:0, 3:3, 4:1}

# lets load the image and take a look
image = cv2.imread(args["image"])
# preprocess the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(src=gray, ksize=(5,5), sigmaX=0)
edged = cv2.Canny(image=blurred, threshold1=75, threshold2=200)

# find contours in the edged image, then initialiize
# the contour that corresponds to the document
cnts = cv2.findContours(image=edged.copy(), mode=cv2.RETR_EXTERNAL,
                        method=cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

docCountours = None

# check if countour(s) are found
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order, keyy top 5 if prsent
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        perimeter = cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(c, epsilon=0.02 * perimeter, closed=True)

        # if the approximated contour has four points
        # then seafely assume that the document has been found
        if len(approx) == 4:
            docCountours = approx

# apply four point perspective trasnformation to both the
# original image and the grayscale image to obtain a
# top-down 'scan' like image
paper = four_point_transform(image, docCountours.reshape(4,2))
warped = four_point_transform(gray, docCountours.reshape(4,2))

# apply Otsu's thresholding to binarize the waraped image
retVal, thresh = cv2.threshold(src=warped, thresh=0, maxval=255,
                               type=cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# find contours in the thresholded image, then initialize
# the list of contours that correspod to questions

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionContours = []

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)  # aspect ratio

    # inorder to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and have
    # an aspect ratio of approximately equal to 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionContours.append(c)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionContours = contours.sort_contours(questionContours,
                                          method="top-to-bottom")[0]  # [0]->get the first element
correct = 0

# each question has 5 possible answers, loop over the questions in
# batches of 5
for (q, i) in enumerate(np.arange(0, len(questionContours), 5)):
    # sort the contours for the current question from
    # left to right, then initialize the index of the
    # bubbled answer
    cnts = contours.sort_contours(questionContours[i:i + 5])[0]
    bubbled = None

    # loop over the sorted contours
    for (j, c) in enumerate(cnts):
        # construct a mask that reveals only the current
        # bubble for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # apply the mask to the thresholded image, then
        # count the  number of non-zero pixels in the
        # bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        # if the current total has a larger number of total
        # non-zero pixels, then we are examining the currently
        # bubbled-in answer
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    # initialize the contour color and the index of the
    # "correct" answer
    color = (0, 0, 255)  # red
    k = ANSWER_KEY[q]

    # check to see if the bubbled-in answer is correct
    if k == bubbled[1]:
        color = (0, 255, 0)  # green
        correct += 1

    # draw the outline of the correct answer on the test
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

# grab the test taker
score = (correct)/5.0 *100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

cv2.imshow("Original Image",image)
cv2.imshow("Graded Test",paper)
cv2.waitKey(0)
cv2.destroyAllWindows()