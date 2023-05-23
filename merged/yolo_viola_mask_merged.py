modelpath = "./yolo.h5"  ## yolo model weights file path

from imageai import Detection
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import sys
import pkg_resources
haar_xml = pkg_resources.resource_filename('cv2', 'data/haarcascade_frontalface_default.xml')
## use full path of haarcascade file if above function doesn't detect it
haar_xml = 'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(haar_xml)
print(faceCascade.empty())     ## checks it xml file is read properly or not
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

yolo = Detection.ObjectDetection()
yolo.setModelTypeAsYOLOv3()
yolo.setModelPath(modelpath)
yolo.loadModel()

cam = cv2.VideoCapture(0) #0=front-cam, 1=external cam
if cam.isOpened() == False:
    print("error opening video file")
    exit(1)
#cam.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
model = load_model('C:\\Users\\Akshay\\Desktop\\IIITB\\2nd SEM\\Visual Recognition\\Project\\Mini project 1\\yolo_&_violajones\\Merged\\densenet121_detection_model.h5')

while True:
    ret, img = cam.read()
    img2, preds = yolo.detectCustomObjectsFromImage(input_image=img, 
                      custom_objects=yolo.CustomObjects(person=True), input_type="array",
                      output_type="array",
                      minimum_percentage_probability=70,
                      display_percentage_probability=True,
                      display_object_name=True) 
    

    for each in preds:
        if each["percentage_probability"] > 70:  ## threshold for accepting person class
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
                im=cv2.resize(img,(224, 224))
                im=im/255.0
                im=np.reshape(im,(1,224,224,3))
                im = np.array(im)
                res=model.predict(im)
                nm = res[0][0] 
                m = res[0][1]
                if m> nm:
                    print("Mask Detected, Door Open")
                    label = 'MASK'
                    box_color = (0, 255, 0)
                elif nm >m:
                    print("No mask Detected, Door closed")
                    label = 'NO MASK'
                    box_color = (255, 0, 0)
    
                cv2.rectangle(img2, (x+x1, y+y1), (x+w+x1, y+h+y1), box_color, 2)
                cv2.rectangle(img2,(x+x1, y+y1-40), (x+w+x1, y+y1), box_color, -1)
                cv2.putText(img2, label, (x+x1, y+y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    cv2.imshow("", img2)
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break
    
cam.release()
cv2.destroyAllWindows()
