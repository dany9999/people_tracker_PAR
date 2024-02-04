import os
import sys

#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

import torch
import pickle
import argparse

import numpy as np

#from data import datamanager

from PIL import Image
from torchvision import transforms

from src.Pedestrian_attribute_Rec.net import build_model
from src.Pedestrian_attribute_Rec.utils2 import read_config



class PAR():
    def __init__(self):
        self.path_config = "src/Pedestrian_attribute_Rec/config/base_extraction.yml"
        self.path_attribute = "src/Pedestrian_attribute_Rec/attributenew.pkl"
        self.path_model = "src/Pedestrian_attribute_Rec/model_best_f1_score.pth"

        config = read_config(self.path_config, False)

        use_gpu = config["n_gpu"] > 0 and torch.cuda.is_available()
        device = torch.device("cuda:0" if use_gpu else "cpu")
        map_location = "cuda:0" if use_gpu else torch.device("cpu")

        self.attribute_name = pickle.load(open(self.path_attribute, "rb"))

        self.model, _ = build_model(config, num_classes=len(self.attribute_name))
        checkpoint = torch.load(self.path_model, map_location=map_location)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model.to(device)
        

    def attribute_recognition(self,image):
        
            image_processing = transforms.Compose(
                [
                    transforms.Resize(size=(256, 192)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            image = image_processing(image)
            image = torch.unsqueeze(image, 0)

            out = self.model(image)
            out = torch.squeeze(out)
            out = torch.sigmoid(out)
            
            out = out.cpu().detach().numpy()
    
            attribute = list(zip(self.attribute_name, out.tolist())) #Ritorna un dizionario coi valori float
            return attribute

    def formatta(self,stringa,prefisso):
    
        # Rimuovi i prefissi
        stringa_tagliata = stringa.replace(prefisso, "")
        return stringa_tagliata


    def select_id_prediction(self, out):
        backpack = out[0][1]
        bag = out[4][1]
        hat = out[2][1]
        male = out[3][1]
        female = out[-1][1]  # Considerando che l'elemento relativo a "personalFemale" è l'ultimo nella lista
        upper_color = out[5]
        lower_color = out[16]
        prefissogiu = "lowerBody"
        prefissosu = "upperBody"
        for idx in range(6, 15):  # Gestione upper_color
            if out[idx][1] > upper_color[1]:
                upper_color = out[idx]

        for idx in range(17, 26):  # Gestione lower_color
            if out[idx][1] > lower_color[1]:
                lower_color = out[idx]

        upper_color = self.formatta(upper_color[0],prefissosu)
        lower_color = self.formatta(lower_color[0],prefissogiu)

        if backpack > 0.35 or bag > 0.35:         # gestione zaino NOTE Threshold da impostare col modello buono
            bag = True
        else:
            bag = False
        
        if hat > 0.3:                          # gestione hat
            hat = True
        else:
            hat = False
        
        if male > female:                       # gestione gender
            gender = "Male"
        else:
            gender = "Female"

        
        lista = [gender,upper_color,lower_color,bag,hat]
        tupla = tuple(lista)
        return tupla
        
    def get_par(self,image):
        attribute = self.attribute_recognition(image)
        tupla = self.select_id_prediction(attribute)
        return tupla
         #ricorda di settare i colori in self.pred



    
        


