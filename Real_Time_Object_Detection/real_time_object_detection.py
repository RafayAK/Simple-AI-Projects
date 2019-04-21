# import necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--confidence', type=float, default=0.2,
                help='minimum probability to filter out weak detections')
args = vars(ap.parse_args())


# define paths to trained caffe model
path_caffemodel = "..//DNN_MODELS//MobileNetSSD_deploy_git.caffemodel"
path_model_architecture = "..//DNN_MODELS//MobileNetSSD_deploy_git.prototxt"

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model form disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(path_model_architecture, path_caffemodel)

# initialize video stream, allow camera to buffer up some frames
print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0) # sleep to buffer frames
fps = FPS().start()


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dims and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843,
                                 (300,300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(detections.shape[2]):
        # extract the probability associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args['confidence']:
            # extract the index of the class label form the 'detections'
            # compute (x, y)-coordinates of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # draw the prediction on the same frame
            label = "{} : {:.2f}".format(CLASSES[idx], confidence *100)
            cv2.rectangle(frame, pt1=(startX, startY), pt2=(endX, endY),
                          color=COLORS[idx], thickness=2)
            y = startY-15 if startY-15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)



    # show the image
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    fps.update()


# do some clean up
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()