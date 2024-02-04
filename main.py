import cv2
import time
import os 
import sys
import yaml
import src.helper_functions as hf

from src.PAR_detector import PAR_detector
from src.Deepsort import DeepSortTracker
#from dataloader import cap
from src.Detection.YoloV5 import YOLOv5Detector
from src.Detection.YOLOv8 import YOLOv8Detector
import json
from src.Dispaly import Display
import argparse

parser = argparse.ArgumentParser(description="Run system")
parser.add_argument("--video", help="video.mp4", required=True)
parser.add_argument("--configuration", help="config", required=True)
parser.add_argument("--results", help="results", required=True)
args = parser.parse_args()

# Logica principale del tuo script

video_filename = args.video
config_filename = args.configuration
results_filename = args.results

# Parameters from config.yml file
with open('./config.yml' , 'r') as f:
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
cap = cv2.VideoCapture(video_filename)

track_history = {}    # Define a empty dictionary to store the previous center locations for each track ID


fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puoi scegliere un altro codec se necessario
output_video = cv2.VideoWriter('results/output_video.avi', fourcc, 15.0, (1920, 1080))  # Imposta la risoluzione desiderata


previous_roi_status = [{}, {}]
people_dict = {}
id_PAR_label = {}
rois = hf.parse_config_file(config_filename, cap)
tt_s = time.perf_counter()
count = 0
fps_period = 2




while cap.isOpened():
          
    success, img = cap.read() # Read the image frame from data source
    if not success:
        break
    current_image = img
    
    if count % fps_period == 0: 
            start_time = time.perf_counter()    #Start Timer - needed to calculate FPS        
            # Object Detection
            
            results = object_detector.run_yolo(current_image)  # run the yolo v5 object detector 
            if len(results) !=0:
                detections , num_objects= object_detector.extract_detections(results, current_image, height=current_image.shape[0], width=current_image.shape[1]) # Plot the bounding boxes and extract detections (needed for DeepSORT) and number of relavent objects detected
                # Object Tracking
                tracks_current = tracker.object_tracker.update_tracks(detections, frame=current_image)

                if count % 50 == 0:   
                    #PAR detection
                    id_PAR_label = par_detector.par_detection(tracks_current, current_image)
                    #tracker.display_track(track_history , tracks_current , img)
                #Display GUI
                display.display_all(tracks_current, track_history,current_image, id_PAR_label, rois, previous_roi_status, count)
            
                #Count metrics for ROI
                people_dict, previous_roi_status = hf.update_people_dict(people_dict, tracks_current, rois, previous_roi_status, cap.get(cv2.CAP_PROP_FPS)/fps_period)
            
            # FPS Calculation
            end_time = time.perf_counter()
            total_time = end_time - start_time
            fps = 1 / total_time
            fps = cap.get(cv2.CAP_PROP_FPS)
        # Descriptions on the output visualization

            cv2.putText(current_image, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            resized_img = cv2.resize(current_image, (1280, 720))
            cv2.imshow('video_show',resized_img)
            output_video.write(current_image)
    
    
       
   

    if cv2.waitKey(1) & 0xFF == 27:
        break
    count = count +1 

# tf = open("results/PAR_pred_duke.json", "w")
# json.dump(par_detector.id_PAR_label, tf, indent= 2)
# tf.close()    
id_PAR_label = par_detector.par_detection(tracks_current, current_image)



# Release and destroy all windows before termination
cap.release()
output_video.release()
cv2.destroyAllWindows()

hf.set_person_attributes(people_dict, par_detector.PAR_common_solution())
hf.create_the_output_file(people_dict, results_filename)




tt_end = time.perf_counter()

print(tt_end - tt_s)






