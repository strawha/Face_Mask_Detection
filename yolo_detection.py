modelpath = "./yolo.h5"

from imageai import Detection
import cv2
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

yolo = Detection.ObjectDetection()
yolo.setModelTypeAsYOLOv3()
yolo.setModelPath(modelpath)
yolo.loadModel()

cam = cv2.VideoCapture(0) #0=front-cam, 1=external cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

while True:
    ret, img = cam.read()

    img, preds = yolo.detectCustomObjectsFromImage(input_image=img, 
                      custom_objects=yolo.CustomObjects(person=True), input_type="array",
                      output_type="array",
                      minimum_percentage_probability=70,
                      display_percentage_probability=True,
                      display_object_name=True)
    
    cv2.imshow("", img)
    ## press q or Esc to quit    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

cam.release()
cv2.destroyAllWindows()
