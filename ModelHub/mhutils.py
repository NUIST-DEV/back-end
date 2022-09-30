import os
import cv2
import numpy as np
import torch

def get_absolute_path(path):
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    return os.path.abspath(path)


# def get_image_tensor(image_name):
#     if image_name is None:
#         return None
#     # check if image exists
#     if not os.path.exists(image_name):
#         return None
#     # read image
#     image = cv2.imread(image_name)
#     # convert to tensor
#     image = torch.from_numpy(image)
#     return image

def get_dir_and_file_name(path):
    return os.path.split(os.path.abspath(path))
