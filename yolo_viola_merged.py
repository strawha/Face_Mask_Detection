modelpath = "./yolo.h5"

from imageai import Detection
import cv2
import os
import sys
import pkg_resources
haar_xml = pkg_resources.resource_filename(
    'cv2', 'data/haarcascade_frontalface_default.xml')

haar_xml = 'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(haar_xml)
print(faceCascade.empty())
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

yolo = Detection.ObjectDetection()
yolo.setModelTypeAsYOLOv3()
yolo.setModelPath(modelpath)
yolo.loadModel()

cam = cv2.VideoCapture(0) #0=front-cam, 1=external cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, img = cam.read()
    img2, preds = yolo.detectCustomObjectsFromImage(input_image=img, 
                      custom_objects=yolo.CustomObjects(person=True), input_type="array",
                      output_type="array",
                      minimum_percentage_probability=80,
                      display_percentage_probability=True,
                      display_object_name=True) 
    #box = cv2.boxPoints(rc)
    
    for each in preds:
        if each["percentage_probability"] > 85:
            rc = each["box_points"]
            x1, y1, x2, y2 = rc[0], rc[1], rc[2], rc[3]
            roi = img[y1:y2, x1:x2]
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)     
            faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.15,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                        )     
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x+x1, y+y1), (x+w+x1, y+h+y1), (0, 255, 0), 2)
            
    
    cv2.imshow("", img)
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break
    
cam.release()
cv2.destroyAllWindows()