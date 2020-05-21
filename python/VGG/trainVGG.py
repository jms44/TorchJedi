import torch
import numpy as np
import cv2
import os
from vgg19 import *
import sys
import vgg19
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
os.chdir(dir_path + "../..") #everything is from TorchJedi root

SETTINGS = {
    "learningRate": 0.001,
    "dataFolder": "/data/",
    "batchSize": 16,
    }

if __name__ == "__main__":
    #Check if data preprocessed already, if not do so
    print(os.listdir("."))
