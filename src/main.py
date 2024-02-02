import cv2
import time
import os 
import sys
import yaml
import helper_functions as hf

from PAR_detector import PAR_detector
from Deepsort import DeepSortTracker
from dataloader import cap
from Detection.YoloV5 import YOLOv5Detector
from Detection.YOLOv8 import YOLOv8Detector
import json
from Dispaly import Display
# Parameters from config.yml file
with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['people_track']['main']

# Add the src directory to the module search path
sys.path.append(os.path.abspath('src'))

# Get YOLO Model Parameter
MODEL_NAME = config['model_name']

# Visualization Parameters
DISP_FPS = config['disp_fps'] 
DISP_OBJ_COUNT = config['disp_obj_count']

object_detector = YOLOv8Detector(model_name=MODEL_NAME)
tracker = DeepSortTracker()
par_detector = PAR_detector()
display = Display()


track_history = {}    # Define a empty dictionary to store the previous center locations for each track ID


fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puoi scegliere un altro codec se necessario
output_video = cv2.VideoWriter('results/output_video.avi', fourcc, 10.0, (1920, 1080))  # Imposta la risoluzione desiderata


previous_roi_status = [{}, {}]
people_dict = {}
rois = hf.parse_config_file("data/config.txt", cap)
tt_s = time.perf_counter()
count = 0
while cap.isOpened():
          
    success, img = cap.read() # Read the image frame from data source
    
    if not success:
        break    
    if count % 1 == 0: 
        start_time = time.perf_counter()    #Start Timer - needed to calculate FPS        
        # Object Detection
        
        results = object_detector.run_yolo(img)  # run the yolo v5 object detector 
        if len(results) !=0:
            detections , num_objects= object_detector.extract_detections(results, img, height=img.shape[0], width=img.shape[1]) # Plot the bounding boxes and extract detections (needed for DeepSORT) and number of relavent objects detected
            # Object Tracking
            tracks_current = tracker.object_tracker.update_tracks(detections, frame=img)

       
            #PAR detection
            id_PAR_label = par_detector.par_detection(tracks_current, img)
            #tracker.display_track(track_history , tracks_current , img)
        
            #Display GUI
            display.display_all(tracks_current, track_history,img, id_PAR_label, rois)
        
            #Count metrics for ROI
            people_dict, previous_roi_status = hf.update_people_dict(people_dict, tracks_current, rois, previous_roi_status, cap.get(cv2.CAP_PROP_FPS))
        
        # FPS Calculation
        end_time = time.perf_counter()
        total_time = end_time - start_time
        fps = 1 / total_time
    
    # Descriptions on the output visualization
    
    
    #if count % 3 == 0:
        cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        # cv2.putText(img, f'MODEL: {MODEL_NAME}', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        # cv2.putText(img, f'TRACKED CLASS: {object_detector.tracked_class}', (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        # cv2.putText(img, f'TRACKER: {tracker.algo_name}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        # cv2.putText(img, f'DETECTED OBJECTS: {num_objects}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        #hf.display_rois(img, rois)
        cv2.imshow('video_show',img)
        output_video.write(img)
    
   
    if cv2.waitKey(1) & 0xFF == 27:
        break
    count = count +1 

# tf = open("results/PAR_pred_duke.json", "w")
# json.dump(par_detector.id_PAR_label, tf, indent= 2)
# tf.close()    




# Release and destroy all windows before termination
cap.release()
output_video.release()
cv2.destroyAllWindows()

hf.set_person_attributes(people_dict, par_detector.PAR_common_solution())
hf.create_the_output_file(people_dict, 'results/results.json')




tt_end = time.perf_counter()

print(tt_end - tt_s)






