# import necessary packages
import numpy as np
import cv2
import os
from imutils.paths import list_files
import argparse
import ffmpeg

# check out the my stackoverflow answer below for this
# https://stackoverflow.com/a/55747773/4699994
def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode


def correct_rotation(frame, rotateCode):
    return cv2.rotate(frame, rotateCode)


# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help='path to directory containing videos')
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter out weak predictions")

# --skip : We don’t need to detect and store every image because
# adjacent frames will be similar. Instead, we’ll skip N frames
# between detections. You can alter the default of 16 using this argument.
ap.add_argument("-s", "--skip", type=int, default=16,
                help="# of frame to skip before applying face detection")

args = vars(ap.parse_args())

# path to caffe model
path_caffemdel = "..//DNN_MODELS//opencv_face_detector.caffemodel"
path_model_architecture = "..//DNN_MODELS//opencv_face_detector.prototxt"


# load model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(path_model_architecture, path_caffemdel)

# store the total number of faces saved from all videos here
total_faces_saved = 0

vs = None  # defined in to loop


# loop over all videos in the directory
for vid_num, video_path  in enumerate(list_files(args["input"])):
    print("[INFO] extracting faces from video #{} ...".format(vid_num))

    # open a pointer to the video file stream and initialize
    # the total number of frame read and save thus far
    vs = cv2.VideoCapture(video_path)
    read = 0

    # check if video requires rotation
    rotateCode = check_rotation(video_path)

    # loop over frames from the video file stream
    while True:
        # grab the frame from the file
        grabbed, frame = vs.read()

        # if frame not grabbed -> end of video
        if not grabbed:
            break

        # check if frame needs to be rotated
        if rotateCode is not None:
            frame = correct_rotation(frame, rotateCode)


        # increment total number of frames read thus far
        read +=1

        # check to see if we should skip this frame
        if read % args["skip"] != 0:
            continue

        # grab the frame dims and construct blob
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain detections
        # and predictions
        net.setInput(blob)
        detections = net.forward()

        # ensure at least oe face was found
        if len(detections) > 0:
            # we're making assumption that each image has only ONE
            # face, so find bounding box with the largest confidence
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detections with the largest confidence also
            # meets our min_confidence test (i.e. filter out weak
            # detections)
            if confidence > args["confidence"]:
                # compute (x, y) - coordinates of the bounding box for
                # the face and extract face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                face = frame[startY:endY, startX:endX]



                # save the cropped face
                p = os.path.sep.join([args['output'],
                                      "{}.png".format(str(total_faces_saved))])
                cv2.imwrite(p, face)
                total_faces_saved += 1
                print("[INFO] save {} to directory".format(total_faces_saved))


# do some clean up
vs.release()
cv2.destroyAllWindows()