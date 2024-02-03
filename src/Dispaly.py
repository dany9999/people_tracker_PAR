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
        self.total_roi1_passages = 0
        self.total_roi2_passages = 0
        pass
    
    def display_all(self, tracks_current, track_history,img, id_PAR_label, rois, previous_roi_status, count):        
        self.display_rois(img, rois)
        cnt = 0
        for track in tracks_current:
            if not track.is_confirmed():
                continue
            cnt += 1
            track_id = track.track_id  
            # Retrieve the current track location(i.e - center of the bounding box) and bounding box
            location = track.to_tlbr()
            bbox = location[:4].astype(int)
            bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            # Retrieve the previous center location, if available
            prev_centers = track_history.get(track_id ,[])
            prev_centers.append(bbox_center)
            track_history[track_id] = prev_centers
            for i, status in enumerate(previous_roi_status):
                if status.get(track_id, False):
                    if i == 0:
                        bbox_color = (255, 0, 0)
                        break
                    else:
                        bbox_color = (0, 255, 0)
                        break
                else:
                    bbox_color = (0, 0, 255)
            self.display_track(img, prev_centers, bbox, track_id, bbox_color)
            if count % 30:
                self.display_PAR(img,id_PAR_label, track_id,bbox)
        self.draw_total_info_rectangle(img, previous_roi_status, cnt)
    
    def display_track(self , img, prev_centers, bbox, track_id, color):
        
            # Draw the track line, if there is a previous center location
            if prev_centers is not None and DISP_TRACKS == True:
                points = np.array(prev_centers, np.int32)
                cv2.polylines(img, [points], False, (51 ,225, 255), 2)

            if DISP_OBJ_TRACK_BOX == True: 
                cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])), color, thickness=2)
                self.display_id(img, track_id, int(bbox[0]), int(bbox[1]), color)


    def display_id(self, img, track_id, x_large, y_large, color_large):
        x_small, y_small, w_small, h_small = x_large, y_large, 30, 40  # Example coordinates, adjust as needed
        color_white = (255, 255, 255)  # White color in BGR format
        # Draw the smaller white rectangle
        cv2.rectangle(img, (x_small, y_small), (x_small + w_small, y_small + h_small), color_white, thickness=cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = color_large  # Use the same color as the larger rectangle

        # Get the size of the text
        text_size = cv2.getTextSize(str(track_id), font, font_scale, font_thickness)[0]

        # Calculate the position to place the text inside the smaller white rectangle
        text_x = x_small + (w_small - text_size[0]) // 2
        text_y = y_small + (h_small + text_size[1]) // 2

        cv2.putText(img, str(track_id), (text_x, text_y), font, font_scale, text_color, font_thickness)

    
    def draw_total_info_rectangle(self, img, previous_roi_status, cnt):      
        box_height = 200
        box_width = 300
        box_x = 0
        box_y = 0
        text_x_offset = 5
        text_y_offset = 5
        color = (255, 255, 255)
        font_scale = 1
        people_in_roi1 = sum([value for value in previous_roi_status[0].values() if value])
        people_in_roi2 = sum([value for value in previous_roi_status[1].values() if value])
        people_in_roi = people_in_roi1 + people_in_roi2
        self.total_roi1_passages += people_in_roi1
        self.total_roi2_passages += people_in_roi2
        cv2.rectangle(img, (0, 0),  (box_width, box_height), color, thickness=cv2.FILLED)
        cv2.putText(img, "People in ROI:{}".format(people_in_roi), (box_x + text_x_offset, box_y + text_y_offset + int(box_height*0.2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)    
        cv2.putText(img, "Total people:{}".format(cnt), (box_x + text_x_offset, box_y + text_y_offset + int(box_height*0.4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)    
        cv2.putText(img, "Passages in ROI 1:{}".format(self.total_roi1_passages), (box_x + text_x_offset, box_y + text_y_offset + int(box_height*0.6)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)    
        cv2.putText(img, "Passages in ROI 2:{}".format(self.total_roi2_passages), (box_x + text_x_offset, box_y + text_y_offset + int(box_height*0.8)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)    


    def draw_person_info_rectangle(self, img, bbox):
        color_new = (255, 255, 255)  # White color in BGR format
        line_height = 20  # Adjust as needed
        text_height = line_height * 5
        left_offset = 30
        x_new = bbox[0] - left_offset
        y_new = bbox[3] + 2
        w_new = bbox[2] - bbox[0] + 2*left_offset
        h_new = text_height        
        cv2.rectangle(img, (x_new, y_new),  (x_new + w_new, y_new + h_new), color_new, thickness=cv2.FILLED)
        return (x_new, y_new, w_new, h_new)


    def write_person_info(self, img, bbox, id_PAR_label, track_id):
        bbox_height = bbox[3]
        font_scale = 0.5
        if id_PAR_label[track_id][-1][0] == 'Male':
            gender_to_print = 'M'
        else:
            gender_to_print = 'F'
        cv2.putText(img, "Gender:{}".format(gender_to_print), (int(bbox[0]), int(bbox[1]) + int(bbox_height*0.2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)    
        if id_PAR_label[track_id][-1][3] == False:
            if id_PAR_label[track_id][-1][4] == False:
                hatbag = "No hat No bag"
            else:
                hatbag = "Hat"
        else:
            if id_PAR_label[track_id][-1][4] == False:
                hatbag = "Bag"
            else:
                hatbag = "Bag Hat"
        cv2.putText(img, hatbag, (int(bbox[0]), int(bbox[1] +  int(bbox_height*0.5))), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
        cv2.putText(img, "U-L:{}-{}".format(id_PAR_label[track_id][-1][1], id_PAR_label[track_id][-1][2]), (int(bbox[0]), int(bbox[1] +  int(bbox_height*0.8))), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
        

    def display_PAR(self,img,id_PAR_label, track_id,bbox):
        if track_id in id_PAR_label and id_PAR_label[track_id]:
            info_rect_bbox = self.draw_person_info_rectangle(img, bbox)   
            self.write_person_info(img, info_rect_bbox, id_PAR_label, track_id)


    def display_rois(self, img, rois):
        for i, roi in enumerate(rois):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 3
            text_color = (0,0,0)   
            cv2.rectangle(img, roi[0], roi[1], text_color, thickness=font_thickness)
            text_position = (roi[0][0] + 20), (roi[0][1] + 35)
            cv2.putText(img, str(i+1), text_position, font, font_scale, text_color, font_thickness)
            
