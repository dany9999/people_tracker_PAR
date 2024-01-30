import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

import torch
import pickle
import argparse

import numpy as np

from data import datamanager

from PIL import Image
from torchvision import transforms

from models import build_model
from utils import read_config



class PAR2():
    def __init__(self):
        self.path_config = "config/base_extraction.yml"
        self.path_attribute = "attributenew.pkl"
        self.path_model = "model_last.pth"

        config = read_config(self.path_config, False)

        use_gpu = config["n_gpu"] > 0 and torch.cuda.is_available()
        device = torch.device("cuda:0" if use_gpu else "cpu")
        map_location = "cuda:0" if use_gpu else torch.device("cpu")

        attribute_name = pickle.load(open(self.path_attribute, "rb"))

        model, _ = build_model(config, num_classes=len(attribute_name))
        checkpoint = torch.load(self.path_model, map_location=map_location)

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(device)

        self.model = model

       

def function(self,image):
    
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

        return out

    

    # result = extractor(args.config, image, 2)
    # result = extractor(path_config=args.config, path_attribute='peta_attribute.pkl', path_model="/content/drive/Shared drives/REID/HIEN/Models/OSNet_Person_Attribute_Refactor/checkpoints/0731_232453/model_best_accuracy.pth", image=image, return_type=0)



