from multiprocessing import Manager, Process
from picamera2 import Picamera2
from image_search.object_center import ObjectCenter
from image_search.pid import PID
from gpiozero import AngularServo
import pkg_resources
import signal
import time
import sys
import cv2

servo_range = (-90, 90)
servo_pan  = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)
servo_tilt = AngularServo(19, min_pulse_width=0.0006, max_pulse_width=0.0023)

# thanks to Adrian Rosebrock whose code this was based on:
# https://pyimagesearch.com/2019/04/01/pan-tilt-face-tracking-with-a-raspberry-pi-and-opencv/

# handle a keyboard interrupt
def signal_handler(sig, frame):
    # print a status message
    print("[INFO] You pressed `ctrl + c`! Exiting...")

    # disable the servos
    servo_pan.close()
    servo_tilt.close()

    # exit
    sys.exit()

def find_object_center(args, obj_x, obj_y, center_x, center_y):
    signal.signal(signal.SIGINT, signal_handler)

    # start the video stream and wait for the camera to warm up
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"format": "XRGB8888"}))
    cam.start()
    time.sleep(1)

    # initialize the object center finder
    obj = ObjectCenter(args["cascade"])

    # loop indefinitely
    while True:
        # grab the frame from the threaded video stream and flip it
        # vertically (since our camera was upside down)
        frame = cam.capture_array()
        frame = cv2.flip(frame, 0)

        # calculate the center of the frame as this is where we will
        # try to keep the object
        (H, W) = frame.shape[:2]
        center_x.value = W // 2
        center_y.value = H // 2

        # find the object's location
        objectLoc = obj.update(frame, (center_x.value, center_y.value))

        if objectLoc is not None:
            ((objX, objY), rect) = objectLoc
            obj_x.value = objX
            obj_y.value = objY

        #     # draw the object on the frame
        #     (x, y, w, h) = rect
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # display the frame to the screen
        # cv2.imshow("Pan-Tilt Face Tracking", frame)
        # cv2.waitKey(1)

def pid_process(output, p, i, d, obj_coord, center_coord):
    # create a PID and initialize it
    p = PID(p, i, d)
    p.initialize()

    # loop indefinitely
    while True:
        # calculate the error
        error = center_coord.value - obj_coord.value

        # update the value
        output.value = p.update(error)

def in_servo_range(val, start, end):
    # determine the input value is in the range start to end
    return (val >= start and val <= end)

def set_servos(pan, tilt):
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    while True:
        pan_angle = -1 * pan.value
        tilt_angle = -1 * tilt.value

        # if the pan angle is within the range, pan
        if in_servo_range(pan_angle, servo_range[0], servo_range[1]):
            pan_to(pan_angle)

        # if the tilt angle is within the range, tilt
        if in_servo_range(tilt_angle, servo_range[0], servo_range[1]):
            tilt_to(tilt_angle + 0)

def pan_to(angle):
    return
    servo_pan.angle = angle

def tilt_to(angle):
    return
    servo_tilt.angle = angle

if __name__ == '__main__':
    haar_path = pkg_resources.resource_filename('cv2', 'data/haarcascade_frontalface_default.xml')

    # initialize a manager to store data between the processes
    with Manager() as manager:
        # set the initial values for the object center and pan/tilt
        center_x = manager.Value('i', 0)
        center_y = manager.Value('i', 0)

        # set the object's (x, y)-coordinates
        obj_x = manager.Value('i', 0)
        obj_y = manager.Value('i', 0)

        # pan and tilt values will be managed by independent PIDs
        pan = manager.Value('i', 0)
        tilt = manager.Value('i', 45)
        pan_to(pan.value)
        tilt_to(tilt.value)

        # set PID values
        pan_p = manager.Value('f', 0.000)
        pan_i = manager.Value('f', 0.00)
        pan_d = manager.Value('f', 0.00000)

        tilt_p = manager.Value('f', 0.00)
        tilt_i = manager.Value('f', 0.00)
        tilt_d = manager.Value('f', 0.0000)

        # we have 4 processes to start:
        # 1. find_object_center - finds the object center
        # 2. panning            - PID control loop determines panning angle
        # 3. tilting            - PID control loop determines tilting angle
        # 4. set_servos         - sets the pan and tilt servos
        processes = [
            Process(target=find_object_center, args=({"cascade": haar_path}, obj_x, obj_y, center_x, center_y)),
            Process(target=pid_process, args=(pan, pan_p, pan_i, pan_d, obj_x, center_x)),
            Process(target=pid_process, args=(tilt, tilt_p, tilt_i, tilt_d, obj_y, center_y)),
            Process(target=set_servos, args=(pan, tilt))
        ]

        # start all processes
        for process in processes:
            process.start()

        # join all processes
        for process in processes:
            process.join()

        # disable the servos
        pan_to(0)
        tilt_to(75)
        servo_pan.close()
        servo_tilt.close()