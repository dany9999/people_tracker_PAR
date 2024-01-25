import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model


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
 

######################################################################
# Argument
# ---------
#parser = argparse.ArgumentParser()
#parser.add_argument('image_path', help='Path to test image')
#parser.add_argument('--dataset', default='market', type=str, help='dataset')
#parser.add_argument('--backbone', default='resnet50', type=str, help='model')
#parser.add_argument('--use-id', action='store_true', help='use identity loss')  #????
#args = parser.parse_args()

#assert args.dataset in ['market', 'duke']
#assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']
dataset = "market" #
#model_name = '{}_nfc_id'.format('resnet50') if False else '{}_nfc'.format(args.backbone) #aggiusta la stringa in base all'uso dell'id.... {}_nfc_id formattaizone stringhe in python
model_name = '{}_nfc'.format('resnet50')  #con id
use_id = False
#num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset] #setto il numero di classi e di id
num_label, num_id = num_cls_dict[dataset], num_ids_dict[dataset]
print("Info model_name: {}# dataset: {} # num_label: {} # num_id: {} # ".format(model_name,dataset,num_label,num_id))

######################################################################
# Model and Data
# ---------
def load_network(network):   #Carico il modello in base al dataset
    save_path = os.path.join('src','Person_Attribute_Recognition_MarketDuke_master','checkpoints', dataset, model_name, 'net_last.pth')
    print(save_path)
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(path):   #carico il path e applico la trasformazione
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src
image_path = "src/Person_Attribute_Recognition_MarketDuke_master/test_sample/test3.jpg"

model = get_model(model_name, num_label, use_id=use_id, num_id=num_id) #from net, la get model carica la backbone
model = load_network(model)
model.eval()  #imposta il modello in eval mode.

src = load_image(image_path)   # carica le immaginiclear

#######input_rete = # COPIAMO IL FRAME DELLA PERSONA
#IMPOSTATE LE VARIABILI IN MODO STATICO


######################################################################
# Inference
# ---------
our_list_duke = [          
        "color of upper-body clothing",
        "wearing hat",
        "color of lower-body clothing",
        "carrying backpack",
        "gender"

]

all_pred = { "color of upper-body clothing":"non_set",              # values set to "non_set" 
        "wearing hat":"not_set",
        "color of lower-body clothing":"not_set",
        "carrying backpack":"not_set",
        "gender":"not_set"}

class predict_decoder(object):

    def __init__(self, dataset):
        with open('src/Person_Attribute_Recognition_MarketDuke_master//doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('src/Person_Attribute_Recognition_MarketDuke_master//doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)


    def decode(self, pred):    #This function print only the charateristics of our interest: gender,bag, hat, upper colour, lower colour
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]  
            if chooce[pred[idx]]:
                all_pred[name] = chooce[pred[idx]]                #set all predicted features in all_pred. NOTE: some features may not be set
                #print('{}: {}'.format(name, chooce[pred[idx]]))

                    
    # def decode(self, pred):    # Original print function
    #     pred = pred.squeeze(dim=0)
    #     for idx in range(self.num_label):
    #         name, chooce = self.attribute_dict[self.label_list[idx]]  
    #         if chooce[pred[idx]]:
    #                 our_pred[name] = chooce[pred[idx]]
    #                 print('{}: {}'.format(name, chooce[pred[idx]]))

    def select_id_prediction(self,all_pred):     #we use backpack_value as representative value of handbag and bag
        bag_value = all_pred["carrying bag"]
        handbag_value = all_pred["carrying handbag"]
        backpack_value = all_pred["carrying backpack"]
        
        if backpack_value == "no":    #if backpack is no, we check if bag_value is yes or handbag_value is yes to set backpack_value to "yes"
            if bag_value == "yes":   
                all_pred["backpack"] = "yes"
            if handbag_value == "yes":
                all_pred["backpack"] = "yes"
    
    def print_pred(self,all_pred,our_list_duke):
        for name in our_list_duke:
            if all_pred[name] != "not_set" :                      #print only the features set
                print("{} : {}".format(name,all_pred[name]))

        

if not use_id:
    out = model.forward(src)
else:
    out, _ = model.forward(src)

pred = torch.gt(out, torch.ones_like(out)/3 )  # change denominator to change the threshold, now is 1/3 = 0.333 
Dec = predict_decoder(dataset) 
Dec.decode(pred)
Dec.select_id_prediction(all_pred)
Dec.print_pred(all_pred,our_list_duke)  # here we print the result of prediction
#print(pred)
