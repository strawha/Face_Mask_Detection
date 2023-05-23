
import cv2
import sys

import pkg_resources
haar_xml = pkg_resources.resource_filename(
    'cv2', 'data/haarcascade_frontalface_default.xml')

# Path of Haar Cascade XML file
faceCascade = cv2.CascadeClassifier(haar_xml)

# To check if xml file was read properly
print(faceCascade.empty())

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

# Press q to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
