from PAR import PAR
from PIL import Image
import cv2
from collections import Counter

class PAR_detector:

    def __init__(self) :
        self.par = PAR()
        self.id_PAR_label = {}
        self.common_solution = {}


    def par_detection(self,tracks_current,img):
        for track in tracks_current:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            location = track.to_tlbr()
            bbox = location[:4].astype(int)
            bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            
            #Create list in dictonary
            if track_id not in self.id_PAR_label.keys(): 
                self.id_PAR_label[track_id] = list()

            try:
                cropped_image = img[bbox[1]:bbox[3], bbox[0]: bbox[2]]
                bgr_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                cropped_image_pil = Image.fromarray(bgr_image)
                self.id_PAR_label[track_id].append(self.par.get_par(cropped_image_pil))
            except:
                pass
                            
        return self.id_PAR_label


    def PAR_common_solution(self):
        self.common_solution = {}

        for key, predictions in self.id_PAR_label.items():
            self.common_solution[key] = self.frequency_PAR(predictions)

        return self.common_solution    

    def frequency_PAR(self, list_prediction):
        
        # Inizializza un dizionario per tenere traccia delle frequenze di ciascun elemento
        frequency = {'gender': Counter(), 'hat': Counter(), 'bag': Counter(),
                    'color_up': Counter(), 'color_low': Counter()}

        # Itera attraverso la lista di tuple e aggiorna le frequenze
        for pred in list_prediction:
            
            frequency['gender'][pred[0]] += 1
            frequency['hat'][pred[1]] += 1
            frequency['bag'][pred[2]] += 1
            frequency['color_up'][pred[3]] += 1
            frequency['color_low'][pred[4]] += 1

        # Inizializza una tupla con i valori più frequenti
        best_freq_pred = (
            frequency['gender'].most_common(1)[0][0],
            frequency['hat'].most_common(1)[0][0],
            frequency['bag'].most_common(1)[0][0],
            frequency['color_up'].most_common(1)[0][0],
            frequency['color_low'].most_common(1)[0][0]
        )

        return best_freq_pred 
