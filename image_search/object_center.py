import imutils
import cv2

class ObjectCenter:
    def __init__(self, haar_path):
        # load OpenCV's Haar cascade face detector
        self.detector = cv2.CascadeClassifier(haar_path)

    def update(self, frame, frame_center):
        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect all faces in the input frame
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=9, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # check to see if a face was found
        if len(rects) > 0:
            # extract the bounding box of the face and compute its
            # center
            (x, y, w, h) = rects[0]
            faceX = int(x + (w / 2.0))
            faceY = int(y + (h / 2.0))

            # return the center (x, y)-coordinates of the face
            return ((faceX, faceY), rects[0])

        # return None if no faces found
        return None
