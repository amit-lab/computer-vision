import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)

# Take lower bound and upper bound of tracking object color
lower_red = np.array([0, 125, 125])
upper_red = np.array([10, 255, 255])

while True:
    ret, frame = cam.read()
    frame = cv.flip(frame, 1)

    w = frame.shape[1]
    h = frame.shape[0]

    # Smooth the image/frame///
    image_smooth  = cv.GaussianBlur(frame, (7,7), 0)

    # Define ROI
    mask = np.zeros_like(frame)

    mask[50:350, 50:350] = [255, 255, 255]

    image_roi = cv.bitwise_and(image_smooth, mask)
    cv.rectangle(frame, (50,50), (350,350), (0,0,255), 2)
    cv.line(frame, (150,50), (150,350), (0,0,255), 1)
    cv.line(frame, (250,50), (250,350), (0,0,255), 1)
    cv.line(frame, (50,150), (350,150), (0,0,255), 1)
    cv.line(frame, (50,250), (350,250), (0,0,255), 1)


    # Threshold image
    image_hsv = cv.cvtColor(image_smooth, cv.COLOR_BGR2HSV)
    image_threshold = cv.inRange(image_hsv, lower_red, upper_red)

    # Find contour
    contours, hairarchy = cv.findContours(image_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Find the index of the largest contours

    if len(contours) != 0:
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        # Pointer on the video
        M = cv.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv.circle(frame, (cx,cy), 4, (0,255,0), -1)

            # Cursor Motion
            if cx in range(150,250):
                if cy < 150:
                    print("Upper Middle")

                if cy > 250:
                    print("Lower Middle")

                else:
                    print("Center")

            if cy in range(150,250):
                if cx < 150:
                    print("Left Middle")

                elif cx > 250:
                    print("Right Middle")

                else:
                    print('Center')


    cv.imshow('Camera', frame)
    k = cv.waitKey(10)
    if k==27:
        break


cam.release()
