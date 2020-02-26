import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)

# Take lower bound and upper bound of tracking object color
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

while True:
    ret, frame = cam.read()

    # Smooth the image/frame///
    image_smooth  = cv.GaussianBlur(frame, (7,7), 0)

    # Threshold image
    image_hsv = cv.cvtColor(image_smooth, cv.COLOR_BGR2HSV)
    image_threshold = cv.inRange(image_hsv, lower_yellow, upper_yellow)

    # Find contour
    contours, hairarchy = cv.findContours(image_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Find the index of the largest contours

    if len(contours) != 0:
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x_bound, y_bound, w_bound, h_bound = cv.boundingRect(cnt)
        cv.rectangle(frame, (x_bound, y_bound), (x_bound+w_bound, y_bound+h_bound), (255,0,0),  3)

    cv.imshow('Camera', frame)
    k = cv.waitKey(10)
    if k==27:
        break


cam.release()
