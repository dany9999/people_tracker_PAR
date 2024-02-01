from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt

with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['people_track']['deep_sort']

#Visualization parameters

DISP_TRACKS = config['disp_tracks']
DISP_OBJ_TRACK_BOX = config['disp_obj_track_box']
OBJ_TRACK_COLOR = tuple(config['obj_tack_color'])
OBJ_TRACK_BOX_COLOR = tuple(config['obj_track_box_color'])

class Display():
    def __init__(self) :
        pass
    
    def display_all(self, tracks_current, track_history,img, id_PAR_label, rois):
        
        self.display_rois(img, rois)
        for track in tracks_current:
            if not track.is_confirmed():
                continue
            track_id = track.track_id  
            # Retrieve the current track location(i.e - center of the bounding box) and bounding box
            location = track.to_tlbr()
            bbox = location[:4].astype(int)
            bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            # Retrieve the previous center location, if available
            prev_centers = track_history.get(track_id ,[])
            prev_centers.append(bbox_center)
            track_history[track_id] = prev_centers

            self.display_track(img, prev_centers, bbox, track_id)
            self.display_PAR(img,id_PAR_label, track_id,bbox)
        
    
    def display_track(self , img, prev_centers, bbox, track_id):
        
            # Draw the track line, if there is a previous center location
            if prev_centers is not None and DISP_TRACKS == True:
                points = np.array(prev_centers, np.int32)
                cv2.polylines(img, [points], False, (51 ,225, 255), 2)

            if DISP_OBJ_TRACK_BOX == True: 
                cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),1)
                cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)


    def display_PAR(self,img,id_PAR_label, track_id,bbox):
        cv2.putText(img, "gender:{}".format(id_PAR_label[track_id][-1][0]), (int(bbox[0]), int(bbox[1] + 25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)    
        cv2.putText(img, "up:{}".format(id_PAR_label[track_id][-1][1]), (int(bbox[0]), int(bbox[1] + 50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(img, "low:{}".format(id_PAR_label[track_id][-1][2]), (int(bbox[0]), int(bbox[1] + 100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(img, "bag:{}".format(id_PAR_label[track_id][-1][3]), (int(bbox[0]), int(bbox[1] + 140)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(img, "hat:{}".format(id_PAR_label[track_id][-1][4]), (int(bbox[0]), int(bbox[1] + 180)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)    

    def display_rois(self, img, rois):
        cv2.rectangle(img, rois[0][0], rois[0][1], (255,0,0), 1)
        cv2.rectangle(img, rois[1][0], rois[1][1], (0,255,0), 1)

