# import necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound


# play an alarm sound
def sound_alarm(path):
    playsound.playsound(path, block=False)  # block=False -> play asynchronously


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x,y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizonatal
    # eye landmark
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio(ear)
    ear = (A + B) / (2 * C)

    # return the eye aspect ratio
    return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=str, default="alarm.wav",
	help="path alarm .WAV file")
args = vars(ap.parse_args())



# define two constants, one for the EAR to indicate
# blink and then a second constant for the number of
# consecutive frames the eye must be below the threshold
# for the alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48

# initialize the frame counter as well as the a bool used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False


# paths to model
path_to_landmark_model = "..\\DNN_MODELS\\shape_predictor_68_face_landmarks.dat"

# initialize dlib's face detector(HOG based) and the facial
# landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_to_landmark_model)

# grab the indices of the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)  # let frames buffer

# loop over the frames
while True:
    # grab the frame from the threaded video file stream, resize
    # it an convert it to gray scale
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in grayscale image
    rects = detector(gray, upsample_num_times=0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Soukupová and Čech recommend averaging both eye aspect
        # ratios together to obtain a better estimation
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for both left and right eye, then
        # visualize each of the eyes with a green outline
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # now check if the person in the frame is dozing off
        if ear < EYE_AR_THRESH:
            # eyes closed
            COUNTER += 1

            # if the eyes were closed for a sufficent number of frames
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    # play alarm, on background thread
                    # t = Thread(target=sound_alarm, args=("alarm.wav",))
                    # t.deamon = True
                    # t.start()
                    sound_alarm(args['alarm'])
                    # draw an alarm on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # otherwise, the EAR is not below the blink thresh i.e. eyes
            # open , so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False

        # draw an counter on the frame
        cv2.putText(frame, "{}".format(COUNTER), (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if key pressed in 'q', break the loop
    if key == ord('q'):
        break


# do some cleanpu
cv2.destroyAllWindows()
vs.stop()