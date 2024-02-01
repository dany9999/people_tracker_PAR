import json
import cv2

class TrackedPerson:
    def __init__(self, id, gender=None, bag=None, hat=None, upper=None, lower=None):
        self.id = id
        self.gender = gender
        self.bag = bag
        self.hat = hat
        self.upper_color = upper
        self.lower_color = lower
        self.roi_passages = [0,0]
        self.roi_persistence_times = [0,0]
        self.roi1passages = 0
        self.roi1_persistence_time = 0
        self.roi2passages = 0
        self.roi2_persistence_time = 0       
    
    def to_dict(self):
        self.roi1passages = self.roi_passages[0]
        self.roi2passages = self.roi_passages[1]
        self.roi1_persistence_time = self.roi_persistence_times[0]        
        self.roi2_persistence_time = self.roi_persistence_times[1]
        custom_dict = {}
        for key, value in vars(self).items():
            if not isinstance(value, list):
                custom_dict[key] = value
        return custom_dict
        

def parse_config_file(roi_file_path, cap):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    with open(roi_file_path, 'r') as file:
        roi_info = json.load(file)
    roi_names = ["roi1", "roi2"]
    rois = []
    for roi_name in roi_names:
        curr_roi_info = roi_info.get(roi_name, {})
        roi_x = int(curr_roi_info.get("x", 0.0) * frame_width)
        roi_y = int(curr_roi_info.get("y", 0.0) * frame_height)
        roi_width = int(curr_roi_info.get("width", 0.0) * frame_width)
        roi_height = int(curr_roi_info.get("height", 0.0) * frame_height)
        roi = ((roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height))
        rois.append(roi)
    return rois


def update_people_dict(people_dict, tracks_current, rois, previous_roi_status, fps):
    for track in tracks_current:
        if not track.is_confirmed():
                continue
        if people_dict.get(track.track_id) is None:
            # So far instantiate only with id, later can be with other tracked attributes
            people_dict[track.track_id] = TrackedPerson(track.track_id)
        people_dict, previous_roi_status = update_passages_persistence(rois, track, people_dict, previous_roi_status, fps)
    return people_dict, previous_roi_status  




def update_passages_persistence(rois, track, people_dict, previous_roi_status, fps):
    # Check if object center is inside ROI1 and update counters
    #print(vars(track))
    curr_person = people_dict[track.track_id]    
    bbox = track.to_tlbr()
    class_id = track.track_id        
    # Calculate object center
    center_x = (bbox[0] + bbox[2]) / 2 
    center_y = (bbox[1] + bbox[3]) / 2 
    # Check if object center is inside ROI1
    for i, roi in enumerate(rois):
        if roi[0][0] < center_x < roi[1][0] and roi[0][1] < center_y < roi[1][1]:
            # Update occurrences for ROI1 and object ID
            curr_person.roi_persistence_times[i] += 1/fps
            # Check if the object is already inside the ROI
            if not previous_roi_status[i].get(class_id, False):
                curr_person.roi_passages[i] += 1
                previous_roi_status[i][class_id] = True
        else:
            if previous_roi_status[i].get(class_id, False):
                previous_roi_status[i][class_id] = False
        
    return people_dict, previous_roi_status


def create_the_output_file(people_dict, file_path):     
    people_list = [value.to_dict() for value in people_dict.values()]
    results_dict = {"people": people_list}
    with open(file_path, 'w') as file:
        json.dump(results_dict, file, indent= 2)
        

