#import necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import argparse



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

args = vars(ap.parse_args())
# define paths to model and its prototxt
model_architecture_path = "..\\DNN_MODELS\\opencv_face_detector.prototxt"
caffe_model_path = '..\\DNN_MODELS\\opencv_face_detector.caffemodel'

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(model_architecture_path, caffe_model_path)

# initialize video stream
print("[INFO] starting video stream from webcam...")
vs = VideoStream(src=0).start()
time.sleep(2.0) # sleep to let frames buffer

# loop over the frames from the video stream
while True:
	# read the next frame from the webcam and resize it
	frame = vs.read()
	#frame = imutils.resize(frame, width=400)  # resize to w=400, maintain asptect ratio

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]  # (H, W, depth)
		print("{}, {}".format(H, W))

	# construct blog from the frame, pass it through the CNN
	# obtain our output predictions, and initialize the list of
	# bounding boxes
	# size=(300,300) required by RESNET implementation, check prototext file
	# mean values provided by author of the RESENT model, using training data
	blob = cv2.dnn.blobFromImage(image=frame, scalefactor=1.0,
								 size=(300, 300), mean=(104.0, 177.0, 123.0))


	net.setInput(blob)
	detections = net.forward()
	rects = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# filter weak detections
		if detections[0, 0, i, 2] > args['confidence']:
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY),
						  (0, 255, 0), 2)

	# update the centroid tracker using computed set of bounding
	# boxes
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw the ID of the object and the centroid of the
		# object on the frame
		text = "ID {}".format(objectID)
		# draw text just over the centroid
		cv2.putText(frame, text, org=(centroid[0] - 10, centroid[1] - 10),
					fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
					color=(0, 255, 0), thickness=2)


		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0),
				   -1)  # thickness=-1 -> filled circle

	# show output
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

# do some cleanup
cv2.destroyAllWindows()
vs.stop()
