import sys
import os
from PIL import Image
import torch
from IModel import IModel
import json

IMAGES_DIR = '../images/'
MODEL_DIR = './models/'
sys.path.append(MODEL_DIR)
# set working directory to file location
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_model(model_dir):
    # check if model exists
    if not os.path.exists(MODEL_DIR + model_dir):
        return None
    # list py file by os
    files = os.listdir(MODEL_DIR + model_dir)
    # find model file
    model_file = None
    for file in files:
        if file.endswith('.py'):
            model_file = file
            break
    if model_file is None:
        return None
    model_name = model_file[:-3]
    sys.path.append(MODEL_DIR + model_dir)
    # import model
    model = __import__(model_name)
    # create model instance
    model = model.__dict__[model_name]
    model = model()
    print('Loaded model: ' + model.name)
    return model


def load_all_models():
    models = {}
    # models placed in individual folders
    for model_dir in os.listdir(MODEL_DIR):
        model = load_model(model_dir)
        if model is not None:
            models[model.name] = model
    return models


def get_image_by_name(image_name):
    if image_name is None:
        return None
    # check if image exists
    path = os.path.abspath(IMAGES_DIR + image_name)
    # print(path)
    if not os.path.exists(path):
        raise Exception('Image not found')
        # return None

    image = Image.open(path)

    return image


def init(image_dir=None):
    # load model
    load_all_models()
    # load variables
    if image_dir is not None:
        global IMAGES_DIR
        IMAGES_DIR = image_dir


class ModelHub:
    def __init__(self):
        # check cuda
        if not torch.cuda.is_available():
            print('CUDA is not available.  No models will be loaded.')
            return

        self.device = torch.device("cuda")
        self.models = load_all_models()
        # for each models, appoint default device
        for model in self.models.values():
            model.device = self.device

    '''Append new models when running'''

    def register(self, name, model: IModel):
        self.models[name] = model

    def get(self, name):
        return self.models[name]

    def get_all(self):
        return self.models

    def get_names(self):
        return self.models.keys()

    def process(self, name, data):
        if name in self.models:
            return self.stringfy(self.models[name].process(data))
        else:
            return self.stringfy({
                "code": 404,
                "error": "model not found"
            })

    def stringfy(self, data):
        if type(data) is str:
            return data
        elif type(data) is dict:
            return json.dumps(data)
        else:
            return json.dumps({
                "code": 500,
                "error": "unknown data type",
                "type": type(data)
            })
