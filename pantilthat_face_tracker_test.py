#!/usr/bin/env python

# Based on article:
# https://core-electronics.com.au/guides/Face-Tracking-Raspberry-Pi/

#Below we are importing functionality to our Code, OPEN-CV, Time, and Pimoroni Pan Tilt Hat Package of particular note.
import cv2, sys, time, os, pkg_resources
from picamera2 import Picamera2
from pantilthat import *

# Load the BCM V4l2 driver for /dev/video0. This driver has been installed from earlier terminal commands.
#This is really just to ensure everything is as it should be.
os.system('sudo modprobe bcm2835-v4l2')
# Set the framerate (not sure this does anything! But you can change the number after | -p | to allegedly increase or decrease the framerate).
os.system('v4l2-ctl -p 40')

# Frame Size. Smaller is faster, but less accurate.
# Wide and short is better, since moving your head up and down is harder to do.
# W = 160 and H = 100 are good settings if you are using and earlier Raspberry Pi Version.
FRAME_W = 640
FRAME_H = 480
CAMERA_VERTICAL_FOV = 62.2
CAMERA_HORIZONTAL_FOV = 48.8

# Default Pan/Tilt for the camera in degrees. I have set it up to roughly point at my face location when it starts the code.
# Camera range is from 0 to 180. Alter the values below to determine the starting point for your pan and tilt.
cam_pan = 0
cam_tilt = 90

# Set up the Cascade Classifier for face tracking. This is using the Haar Cascade face recognition method with LBP = Local Binary Patterns.
# Seen below is commented out the slower method to get face tracking done using only the HAAR method.
# cascPath = 'haarcascade_frontalface_default.xml' # sys.argv[1]
haar_path = pkg_resources.resource_filename('cv2', 'data/haarcascade_frontalface_default.xml')
print(haar_path)
faceCascade = cv2.CascadeClassifier(haar_path)

# Start and set up the video capture with our selected frame size. Make sure these values match the same width and height values that you choose at the start.
cam = Picamera2()
cam.configure(cam.create_video_configuration(main={"format": "XRGB8888", "size": (FRAME_W, FRAME_H)}))
cam.start()
time.sleep(1)

# Turn the camera to the Start position (the data that pan() and tilt() functions expect to see are any numbers between -90 to 90 degrees).
pan(cam_pan)
tilt(cam_tilt)

#Below we are creating an infinite loop, the system will run forever or until we manually tell it to stop (or use the "q" button on our keyboard)
while True:

    # Capture frame-by-frame
    #ret, frame = cap.read()
    frame = cam.capture_array()
    # This line lets you mount the camera the "right" way up, with neopixels above
    frame = cv2.flip(frame, 0)

    # Convert to greyscale for easier faster accurate face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Do face detection to search for faces from these captures frames
    faces = faceCascade.detectMultiScale(gray)
    if (len(faces)) > 0: print(f'\n---- Found {len(faces)} faces ----')

    #Below draws the rectangle onto the screen then determines how to move the camera module so that the face can always be in the centre of screen.

    for (x, y, w, h) in faces:
        # Draw a green rectangle around the face (There is a lot of control to be had here, for example If you want a bigger border change 4 to 8)
        #cv2.rectangle(frame, (x, y), (x, w, y, h), (0, 255, 0), 4)

        # Track face with the square around it
        print((x, y, w, h))

        # Get the centre of the face
        x = x + (w/2)
        y = y + (h/2)
        print('x', x, ',\ty', y)

        # convert to scale of -0.5 to 0.5 relative to frame
        relative_x = float(x / FRAME_W) - 0.5
        relative_y = float(y / FRAME_H) - 0.5
        print('relative x,y: ',relative_x,relative_y)

        # find the angle in degrees relative to camera's FOV
        angle_horizontal = relative_x * CAMERA_HORIZONTAL_FOV
        angle_vertical = relative_y * CAMERA_VERTICAL_FOV
        print('angle horiz/vert: ', angle_horizontal,angle_vertical)

        # since the servos have a range of -90 to +90, we want
        # to move them from the current position, to the current
        # position plus the angle found by the camera
        cam_pan = get_pan() + angle_horizontal
        cam_tilt = get_tilt() + angle_vertical

        print('Cur Pan :', get_pan(), '\t Tilt: ', get_tilt())
        print('New Pan: ', cam_pan, ',\t Tilt: ', cam_tilt)

        # clamp to range of servos
        cam_pan = max(-90, min(90,cam_pan))
        cam_tilt = max(-90, min(90,cam_tilt))

        # Update the servos
        pan(int(cam_pan))
        tilt(int(cam_tilt))

    time.sleep(0.5)
    #Orientate the frame so you can see it.
    #frame = cv2.flip(frame, 1)

    # Display the video captured, with rectangles overlayed
    # onto the Pi desktop
    #cv2.imshow('f', frame)

    #If you type q at any point this will end the loop and thus end the code.
    #if cv2.waitKey(10) & 0xFF == ord('q'):
    #    break

# When everything is done, release the capture information and stop everything
video_capture.release()
cv2.destroyAllWindows()

