import sys
from IModel import IModel
import torch
import mhutils
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from network_files import FasterRCNN, AnchorsGenerator
from backbone import BackboneWithFPN, LastLevelMaxPool
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

class MyFasterRCNN(IModel):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda')
        _dir, _file = mhutils.get_dir_and_file_name(__file__)
        print(os.path.join(_dir, 'model.pth'))
        self.model = torch.load(os.path.join(_dir, 'model.pth'))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def process(self, data):
        image = self.transform(data).to(self.device)
        image = image.unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(image)
        return prediction


# if __name__ == '__main__':
#     model = FasterRCNN()
#     from PIL import Image
#     prediction = model.process(Image.open('00041030.jpg'))
#     print(prediction)