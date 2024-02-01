from PAR import PAR
from PIL import Image
import cv2


class PAR_detector:

    def __init__(self) :
        self.par = PAR()
        self.id_PAR_label = {}


    def par_detection(self,tracks_current,img):
        for track in tracks_current:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            location = track.to_tlbr()
            bbox = location[:4].astype(int)
            bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            # # Controllo dei limiti del bounding box
            # ymin = max(0, bbox[1])
            # xmin = max(0, bbox[0])
            # ymax = min(img.shape[0], bbox[3])
            # xmax = min(img.shape[1], bbox[2])

            #Create list in dictonary
            if track_id not in self.id_PAR_label.keys(): 
                self.id_PAR_label[track_id] = list()

            # Riconoscimento degli attributi PAR sull'immagine ritagliata
            #if len(self.id_PAR_label[track_id]) < 10:
            try:
                    cropped_image = img[bbox[1]:bbox[3], bbox[0]: bbox[2]]
                    cropped_image_pil = Image.fromarray(cropped_image)
                    self.id_PAR_label[track_id].append(self.par.get_par(cropped_image_pil))
            except:
                    pass
                        
            # cv2.putText(img, "gender:{}".format(self.id_PAR_label[track_id][0][0]), (int(bbox[0]), int(bbox[1] + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)    
            # cv2.putText(img, "up:{}".format(self.id_PAR_label[track_id][0][1]), (int(bbox[0]), int(bbox[1] + 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            # cv2.putText(img, "low:{}".format(self.id_PAR_label[track_id][0][2]), (int(bbox[0]), int(bbox[1] + 45)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            # cv2.putText(img, "bag:{}".format(self.id_PAR_label[track_id][0][3]), (int(bbox[0]), int(bbox[1] + 55)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            # cv2.putText(img, "hat:{}".format(self.id_PAR_label[track_id][0][4]), (int(bbox[0]), int(bbox[1] + 65)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        return self.id_PAR_label

    
