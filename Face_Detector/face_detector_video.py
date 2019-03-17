# This python script applies face detection to videos,
# video streams, and webcams

# import packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the args
# optional arg is --confidence, which is minimum thresh
# we want out NN to fulfill to detect a face
ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help='minimum probability to filter weak detections')

args = vars(ap.parse_args())


# model_architecture_path is the ".prototxt" file with defines the model architecture(i.e layers)
model_architecture_path = "DNN_MODELS\\opencv_face_detector.prototxt"

# caffe_model_path is the ".caffemodel" file which contains the trained weights for the layers
caffe_model_path = 'DNN_MODELS\\opencv_face_detector.caffemodel'

# load model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(model_architecture_path, caffe_model_path)

# initialize video stream from webcam
print("[INFO] starting video stream...")
# a VideoStream  object specifying camera with index zero as the
# source (in general this would be your laptop’s built in
# camera or your desktop’s first camera detected).
#
# A few quick notes here:
#
# Raspberry Pi + picamera users can replace Line 48 with
# vs = VideoStream(usePiCamera=True).start()
#
# if you wish to use the Raspberry Pi camera module.
#
# If you to parse a video file (rather than a video stream)
#  swap out the VideoStream  class for FileVideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)  # warm up for 2 seconds allow frames to be buffered

# --------------------------------------------------------------
# from here loop over frames
while True:
    # grab a frame from the threaded video stream and resize it to have
    # to have a max width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(image=cv2.resize(src=frame, dsize=(300, 300)),
                                 scalefactor= 1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))

    # pass the blob though the NN and get the possible candidates for detections
    # and their predictions
    net.setInput(blob)
    detections = net.forward()

    # now loop over the detections to filter out the ones below thresh
    for i in range(0, detections.shape[2]):
        # extract the confidence associated with i'th prediction
        confidence = detections[0,0,i,2]

        # filter the detections
        if confidence < args["confidence"]:
            continue

        # compute the x,y - coordinates of the bounding boxes for the objects
        box = detections[0,0,i, 3:7] * np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with associated probability
        text = "{:.2f}%".format(confidence*100)
        y = startY - 10 if startY -10 > 10 else startY +10
        cv2.rectangle(img=frame, pt1=(startX, startY), pt2=(endX, endY),
                      color=(0, 0, 255), thickness=2)
        cv2.putText(img=frame, text=text, org=(startX, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0, 0, 255),
                    thickness=2)

    # show the output frame
    cv2.imshow("Frame", frame)

    # waitKey outputs 32bit unicode char,
    # we need 8bit ASCII char to check
    # 0xFF is 8-bit -> 11111111
    # cv2.waitKey(1) & 0xFF -> returns last 8-bits of the 32-bit code
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do some cleanup
cv2.destroyAllWindows()
vs.stop()
