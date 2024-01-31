
# import os
# from net import get_model
# import json
# import torch
# import argparse
# from PIL import Image
# from torchvision import transforms as T

# ######################################################################
# # Settings
# # ---------
# dataset_dict = {
#     'market'  :  'Market-1501',
#     'duke'  :  'DukeMTMC-reID',
# }
# num_cls_dict = { 'market':30, 'duke':23 } # indica il numero di classi per dataset
# num_ids_dict = { 'market':751, 'duke':702 } #specifica quanti individui unici sono presenti in ciascun dataset. Ad esempio, per il dataset 'market', ci sono 751 identità, e per 'duke' ci sono 702 identità.

# transforms = T.Compose([                   #Trasformazioni da applicare all'input
#     T.Resize(size=(288, 144)),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


# our_list_duke = [                           #si puo togliere
#         "color of upper-body clothing",
#         "wearing hat",
#         "color of lower-body clothing",
#         "carrying backpack",
#         "gender",
#         "carrying bag",
#         "carrying handbag"

# ]



# class PAR(object):

#     def __init__(self, dataset):
#         ######################################################################
#         # Argument
#         # ---------
#         self.dataset = dataset
#         #model_name = '{}_nfc_id'.format('resnet50') if False else '{}_nfc'.format(args.backbone) #aggiusta la stringa in base all'uso dell'id.... {}_nfc_id formattaizone stringhe in python
#         self.model_name = '{}_nfc'.format('resnet50')  #con id
#         self.use_id = False
#         #num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset] #setto il numero di classi e di id
#         num_label, num_id = num_cls_dict[self.dataset], num_ids_dict[self.dataset]
#         print("Info model_name: {}# dataset: {} # num_label: {} # num_id: {} # ".format(self.model_name,self.dataset,num_label,num_id))

#         self.all_pred = { "color of upper-body clothing":None,              # si puo togliere 
#             "wearing hat":None,
#             "color of lower-body clothing":None,
#             "carrying backpack":None,
#             "gender":None,
#             "carrying bag": None ,
#             "carrying handbag " : None
#             }
#         self.selected_pred = {"gender": None, "bag": None, "hat": None, "upper_color" : None, "lower_color": None}

#         self.model = get_model(self.model_name, num_label, use_id=self.use_id, num_id=num_id) #from net, la get model carica la backbone
#         self.model = self.load_network(self.model)
#         self.model.eval()  #imposta il modello in eval mode.

#         with open('src/doc/label.json', 'r') as f:
#             self.label_list = json.load(f)[self.dataset]
#         with open('src/doc/attribute.json', 'r') as f:
#             self.attribute_dict = json.load(f)[self.dataset]
#         self.num_label = len(self.label_list)
#         self.result_dict = {}                       #dict risultante definitivo


#     ######################################################################
#     # Model and Data
#     # ---------
#     def load_network(self,network):   #Carico il modello in base al dataset
#         save_path = os.path.join('src','checkpoints', self.dataset, self.model_name, 'net_last.pth')
#         print(save_path)
#         network.load_state_dict(torch.load(save_path))
#         print('Resume model from {}'.format(save_path))
#         return network

#     def load_image(self,image):   #carico il path e applico la trasformazione
#         src = transforms(image)
#         src = src.unsqueeze(dim=0)
#         return src
        


# ######################################################################
# # Inference
# # ---------

#     def attribute_recognition(self,image):
        
#         src = self.load_image(image)   # carica le immaginiclear
#         if not self.use_id:
#             out = self.model.forward(src)
#         else:
#             out, _ = self.model.forward(src)

#         pred = torch.gt(out, torch.ones_like(out)*0.5)  # change denominator to change the threshold, now is 1/3 = 0.333 
#         #Dec = PAR(self.dataset)
#         self.decode(pred)
#         self.select_id_prediction()
#         #self.print_pred(our_list_duke)  # here we print the result of prediction
        
        
#         for idx in range(self.num_label):
#             name = self.label_list[idx]
#             self.result_dict[name] = out[0, idx].item()  # Aggiungere il valore da out al dizionario
        
#         return self.result_dict
    


# ### Metodi supporto inferenza
#     def decode(self, pred):    #This function print only the charateristics of our interest: gender,bag, hat, upper colour, lower colour
#         pred = pred.squeeze(dim=0)
#         for idx in range(self.num_label):
#             name, chooce = self.attribute_dict[self.label_list[idx]]  
#             if chooce[pred[idx]] is not None:
#                 if name in our_list_duke:
#                     self.all_pred[name] = chooce[pred[idx]]                #set all predicted features in all_pred. NOTE: some features may not be set
            
        

#     def select_id_prediction(self):     #we use backpack_value as representative value of handbag and bag
#         bag_value = self.all_pred["carrying bag"]
#         handbag_value = self.all_pred["carrying handbag"]
#         backpack_value = self.all_pred["carrying backpack"]
        

#         # set fields of selected_pred

#         if backpack_value == "yes":    #if backpack is no, we check if bag_value is yes or handbag_value is yes to set backpack_value to "yes"
#             self.selected_pred["bag"] = True
#         elif bag_value == "yes":   
#             self.selected_pred["bag"] = True
#         elif handbag_value == "yes":
#             self.selected_pred["bag"] = True

#         else:
#             if(backpack_value is not None or bag_value is not None or handbag_value is not None):
#                 self.selected_pred["bag"] = False
#             else:
#                 self.selected_pred["bag"] = None

#         if self.all_pred["gender"] is not None:
#             self.selected_pred["gender"] = self.all_pred["gender"]
        
#         if self.all_pred["wearing hat"] is not None:
#             if self.all_pred["wearing hat"] == "yes":
#                 self.selected_pred["hat"] = True
#             else:
#                 self.selected_pred["hat"] = False

#         if self.all_pred["color of upper-body clothing"] is not None:
#             self.selected_pred["upper_color"] = self.all_pred["color of upper-body clothing"]

#         if self.all_pred["color of lower-body clothing"] is not None:
#             self.selected_pred["lower_color"] = self.all_pred["color of lower-body clothing"]
        


#     def print_pred(self,our_list_duke):
#         for name in our_list_duke:
#             if self.all_pred[name] != None :                      #print only the features set
#                 print("{} : {}".format(name,self.all_pred[name]))


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

from Pedestrian_attribute_Rec.net import build_model
from Pedestrian_attribute_Rec.utils2 import read_config



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
        
        #self.model = model
        

   

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
            
            #out[out > 0.7] = 2
            #out[out <= 0.3] = 0
            #out[(out <= 0.7) & (out >= 0.3)] = 1
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

    # result = extractor(args.config, image, 2)
    # result = extractor(path_config=args.config, path_attribute='peta_attribute.pkl', path_model="/content/drive/Shared drives/REID/HIEN/Models/OSNet_Person_Attribute_Refactor/checkpoints/0731_232453/model_best_accuracy.pth", image=image, return_type=0)


    
        


