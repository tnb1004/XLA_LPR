import cv2
import imutils
import numpy as np

cam = cv2.VideoCapture(0)
while True:
    #Capture
    ret, frame = cam.read()
    resized_frame = imutils.resize(frame, width=1000)
    cv2.imshow('RC', resized_frame)
    #'q' button is set to stop the cam feed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()   
