"""
Stuff to figure out-
1- how to determine HSV color range
2- see the contours image
3- compare different contour methods
"""

from collections import deque #primary data structure used to store previous centroid coordinates
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser() #initialize argument parser object
#add arguments
ap.add_argument("-v", "--video", help="path to the (optional) video file\n if not included, webcam will be used")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
#make a dictionary of the supplied CLI arguments
#keys are the name of the argument, values are the values
args = vars(ap.parse_args())


# define the lower and upper boundaries of the object in HSV
# need to find a better way to find these
"""
# light green
colorLower = (29,86,6)
colorUpper = (64,255,255)
"""

colorLower = (109,198,5)
colorUpper = (179,246,240)

# initialize the list of tracked points
# deque has super fast removal and addition times at both ends
pts = deque(maxlen=args["buffer"])


# if a video path was not supplied, grab the reference to the webcam
# 0 is the source of the first camera attatched to a computer
if not args.get("video", False): #if video flag was not false, i.e. it was supplied, we use the webcam
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
	
# allow the camera or video file to warm up
time.sleep(2.0)


while True:
    # grab the current frame
    frame = vs.read()
    # handle the frame from VideoCapture or VideoStream
    if args.get("video", False): #using webcam- VideoCapture
        frame = frame[1]
    else: #using video- VideoStream
        frame
    # if we are viewing a video and we did not grab a frame, we have reached the end of the video
    if frame is None:
        break
    #if we have reached this point, stream/capture is still going
    # resize the frame, blur it, and convert it to HSV
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #this section removes unwanted blobs in our object, and background noise
    mask = cv2.inRange(hsv, colorLower, colorUpper) #constructs mask for the color
    mask = cv2.erode(mask, None, iterations=2) #erosion wears away the borders of the color slightly
    mask = cv2.dilate(mask, None, iterations=2) #increases area of our object again


    # find contours in the mask and initialize the current coordinates of the ball
    # figure out more about how this works
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #returns list of contours. each contour is a numpy array of boundary points
    cnts = imutils.grab_contours(cnts) #returns them as tuples

    
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask by area, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c) #returns coordinates of the centroid and the radius
        M = cv2.moments(c) #M is a dictionary of all moment values calculated- basically just the centroid
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) #formula for computing the centroid's coordinates
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2) #outer circle
            cv2.circle(frame, center, 5, (0, 0, 255), -1) #center
    # update the points queue with the new point
    pts.appendleft(center)

    # loop over the set of previous points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5) #thickness of each segment decreases
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness) #draws the line segment
        # show the frame to our screen
    cv2.imshow("Frame", frame)
    #cv2.imshow("mask", mask)
    key = cv2.waitKey(1) & 0xFF #1 is milliseconds between frames
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file (we are using webcam), stop the camera video stream
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera
else:
    vs.release()
# close all windows
cv2.destroyAllWindows()
