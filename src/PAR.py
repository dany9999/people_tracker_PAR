
import os
from net import get_model
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T

######################################################################
# Settings
# ---------
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
num_cls_dict = { 'market':30, 'duke':23 } # indica il numero di classi per dataset
num_ids_dict = { 'market':751, 'duke':702 } #specifica quanti individui unici sono presenti in ciascun dataset. Ad esempio, per il dataset 'market', ci sono 751 identità, e per 'duke' ci sono 702 identità.

transforms = T.Compose([                   #Trasformazioni da applicare all'input
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#image_path = "test_sample/test3.jpg" # sarebbe image imput

our_list_duke = [          
        "color of upper-body clothing",
        "wearing hat",
        "color of lower-body clothing",
        "carrying backpack",
        "gender"

]



class PAR(object):

    def __init__(self):
        ######################################################################
        # Argument
        # ---------
        self.dataset = "duke" 
        #model_name = '{}_nfc_id'.format('resnet50') if False else '{}_nfc'.format(args.backbone) #aggiusta la stringa in base all'uso dell'id.... {}_nfc_id formattaizone stringhe in python
        self.model_name = '{}_nfc'.format('resnet50')  #con id
        self.use_id = False
        #num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset] #setto il numero di classi e di id
        num_label, num_id = num_cls_dict[self.dataset], num_ids_dict[self.dataset]
        print("Info model_name: {}# dataset: {} # num_label: {} # num_id: {} # ".format(self.model_name,self.dataset,num_label,num_id))

        self.all_pred = { "color of upper-body clothing":"non_set",              # values set to "non_set" 
            "wearing hat":"not_set",
            "color of lower-body clothing":"not_set",
            "carrying backpack":"not_set",
            "gender":"not_set"}
        self.model = get_model(self.model_name, num_label, use_id=self.use_id, num_id=num_id) #from net, la get model carica la backbone
        self.model = self.load_network(self.model)
        self.model.eval()  #imposta il modello in eval mode.

        with open('src/doc/label.json', 'r') as f:
            self.label_list = json.load(f)[self.dataset]
        with open('src/doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[self.dataset]
        self.num_label = len(self.label_list)

    ######################################################################
    # Model and Data
    # ---------
    def load_network(self,network):   #Carico il modello in base al dataset
        save_path = os.path.join('src','checkpoints', self.dataset, self.model_name, 'net_last.pth')
        print(save_path)
        network.load_state_dict(torch.load(save_path))
        print('Resume model from {}'.format(save_path))
        return network

    def load_image(self,image):   #carico il path e applico la trasformazione
        src = transforms(image)
        src = src.unsqueeze(dim=0)
        return src
        


######################################################################
# Inference
# ---------

    def attribute_recognition(self,image):
        # da inserire codice main inferenza
        src = self.load_image(image)   # carica le immaginiclear
        if not self.use_id:
            out = self.model.forward(src)
        else:
            out, _ = self.model.forward(src)

        pred = torch.gt(out, torch.ones_like(out)/3 )  # change denominator to change the threshold, now is 1/3 = 0.333 
        #Dec = PAR(self.dataset) 
        self.decode(pred)
        self.select_id_prediction()
        self.print_pred(our_list_duke)  # here we print the result of prediction
        #print(pred)
        return self.all_pred
    


### Metodi supporto inferenza
    def decode(self, pred):    #This function print only the charateristics of our interest: gender,bag, hat, upper colour, lower colour
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]  
            if chooce[pred[idx]]:
                self.all_pred[name] = chooce[pred[idx]]                #set all predicted features in all_pred. NOTE: some features may not be set
                #print('{}: {}'.format(name, chooce[pred[idx]]))

    def select_id_prediction(self):     #we use backpack_value as representative value of handbag and bag
        bag_value = self.all_pred["carrying bag"]
        handbag_value = self.all_pred["carrying handbag"]
        backpack_value = self.all_pred["carrying backpack"]
        
        if backpack_value == "no":    #if backpack is no, we check if bag_value is yes or handbag_value is yes to set backpack_value to "yes"
            if bag_value == "yes":   
                self.all_pred["backpack"] = "yes"
            if handbag_value == "yes":
                self.all_pred["backpack"] = "yes"

    def print_pred(self,our_list_duke):
        for name in our_list_duke:
            if self.all_pred[name] != "not_set" :                      #print only the features set
                print("{} : {}".format(name,self.all_pred[name]))