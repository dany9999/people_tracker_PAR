import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['people_track']['detector']

TRACKED_CLASS = config['tracked_class']
DOWNSCALE_FACTOR = config['downscale_factor']
CONFIDENCE_THRESHOLD = config['confidence_threshold']
DISP_OBJ_DETECT_BOX = config['disp_obj_detect_box']

class YOLOv8Detector(): 

    def __init__(self, model_name):

        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: " , self.device)

        self.downscale_factor = DOWNSCALE_FACTOR  # Reduce the resolution of the input frame by this factor to speed up object detection process
        self.confidence_threshold = CONFIDENCE_THRESHOLD # Minimum theshold for the detection bounding box to be displayed
        self.tracked_class = TRACKED_CLASS

    def load_model(self , model_name):  # Load a specific yolo v5 model or the default model

        return YOLO(model_name)
 
    def run_yolo(self , frame): 
        self.model.to(self.device) # Transfer a model and its associated tensors to CPU or GPU
        
        yolo_result = self.model.predict(frame, conf=CONFIDENCE_THRESHOLD, classes = 0 , show_labels=False, show_conf= False, show_boxes= False)     # return a list of Results objects

        # Process results list
        for result in yolo_result:
            boxes = result.boxes  # Boxes object for bbox outputs

        
        return boxes
        

    def class_to_label(self, x):

        return self.classes[int(x)]
        
    def extract_detections(self, boxes, frame, height, width):
        
        
        detections = []         # Empty list to store the detections later 
        class_count = 0         # Initialize class count for the frame 
        num_objects = len(boxes)   #extract the number of objects detected
        x_shape, y_shape = width, height
        conf = boxes.conf.tolist()[0]
        boxes = boxes.xyxy.tolist()

        for box in boxes:
            
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                      
            if DISP_OBJ_DETECT_BOX: 
                self.plot_boxes(x1 , y1 , x2 , y2 , frame)
            x_center = x1 + ((x2-x1)/2)
            y_center = y1 + ((y2 - y1) / 2)
            #conf_val = float(box[0][4].item())
            feature = self.tracked_class

            class_count+=1
                    
            detections.append(([x1, y1, int(x2-x1), int(y2-y1)], conf, feature))
            # We structure the detections in this way because we want the bbs expected to be a list of detections in the tracker, each in tuples of ( [left,top,w,h], confidence, detection_class) - Check deep-sort-realtime 1.3.2 documentation

        return detections , class_count
    
    def plot_boxes(self , x1 , y1 , x2 , y2 , frame):  
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

