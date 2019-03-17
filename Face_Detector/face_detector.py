# import libraries
import numpy as np
import cv2
import argparse  # to pass arguments in a cooler way

# construct the argument parse and parse the args
#
# We have one required arguments:
# --image : The path to the input image.
#
# optional arg is --confidence, which is minimum thresh
# we want out NN to fulfill to detect a face
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help='minimum probability to filter weak detections')

args = vars(ap.parse_args())

# now lets, load the model and create blobs from input image

print("[INFO] loading model...")
# model_architecture_path is the ".prototxt" file with defines the model architecture(i.e layers)
model_architecture_path = "..\\DNN_MODELS\\opencv_face_detector.prototxt"

# caffe_model_path is the ".caffemodel" file which contains the trained weights for the layers
caffe_model_path = '..\\DNN_MODELS\\opencv_face_detector.caffemodel'

# load downloaded serialized model
net = cv2.dnn.readNetFromCaffe(model_architecture_path, caffe_model_path)

print(net)

# load the input image and construct and input blob for the image
# by resizing to the fixed 300x300 pixels size required by the model
# and normalizing the pixel values

image = cv2.imread(args["image"])
(h,w) = image.shape[:2]
#blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300), 1.0, (300,300), (104.0, 177.0, 123.0)))
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))


# Next, apply face detection
# pass the blob through the network to get detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()


# From here, go over all the possible candidates for the face and output the one with the highest probability

# loop over the detections
for i in range(0, detections.shape[2]):
    # get probability associated with this prediction
    confidence = detections[0,0,i,2]

    # filter out weak detections using out defined thresh
    if confidence > args["confidence"]:
        # great found a face!

        # compute (x,y) coordinates of the bounding box for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence*100)
        y = startY -10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img=image, pt1=(startX,startY), pt2=(endX,endY),
                      color=(0,0,255), thickness=2)
        cv2.putText(img=image, text=text, org=(startX, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0,0,255),
                    thickness=2)

# show the output image
cv2.imshow("output", image)
cv2.waitKey(0)
