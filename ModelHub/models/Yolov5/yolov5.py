import sys
from IModel import IModel
import torch
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json


# convert results.xyxyn to json
def ConvertTensorToJSON(results):
    tensor = results.xyxyn[0].cpu().detach().numpy().tolist()
    arr = []
    for i in range(len(tensor)):
        arr.append({
            "x1": tensor[i][0],
            "y1": tensor[i][1],
            "x2": tensor[i][2],
            "y2": tensor[i][3],
            "confidence": tensor[i][4],
            "class": int(tensor[i][5]),
            "verbose": results.names[int(tensor[i][5])]
        })
    return json.dumps(arr)

class yolov5(IModel):
    def __init__(self, device='cuda') -> None:
        super().__init__()
        self.name = "Yolov5"
        self.alias = "Yolov5"
        self.description = "Yolov5"
        #get current file path
        self.path = os.path.dirname(os.path.realpath(__file__))
        print(self.path)
        # './ultralytics_yolov5_master'
        self.device = torch.device(device)
        self.model = torch.hub.load(os.path.join(self.path,'./ultralytics_yolov5_master/'), 
                                    'custom',
                                    source='local',
                                    # use custom pretrained weights
                                    path=os.path.join(self.path,'yolov5s.pt'),
                                    ).to(self.device)

    def predict(self, image):
        results = self.model(image)
        return results

    def process(self, data):
        if self.device is None:
            return {'error': 'No device selected!'}
        if data is None:
            return {'error': 'No data received!'}
        if type(data) is not dict:
            return {'error': 'Data is not a dict!'}
        if "image" not in data.keys():
            return {'error': 'No image in data!'}
        image = data["image"].to(self.device)

        results = self.model(image)
        return ConvertTensorToJSON(results)