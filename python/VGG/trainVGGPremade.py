import torch
import cv2
import os
import sys
import os
import glob
import random
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path + "/../..")


def convertImage(path, outPath):
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 512))
    cv2.imwrite(outPath, img)


print("Starting Data Loading...")
if(not os.path.isfile("./data/polarLows/processed/trainingDataX.pt")):
    #we must create directory and fill it
    if(not os.path.isdir("./data/polarLows/processed")):
        os.mkdir("./data/polarLows/processed")
        os.mkdir("./data/polarLows/processed/positive")
        os.mkdir("./data/polarLows/processed/negative")
        i = 0
        for path in glob.glob("./data/polarLows/raw/positive/*.jpeg"):
            convertImage(path, "./data/polarLows/processed/positive/" + str(i).zfill(4) + ".jpeg")
            i+=1

        i = 0
        for path in glob.glob("./data/polarLows/raw/negative/*.jpeg"):
            convertImage(path, "./data/polarLows/processed/negative/" + str(i).zfill(4) + ".jpeg")
            i+=1

        #now time to create big tensor
    positives = glob.glob("./data/polarLows/processed/positive/*.jpeg")
    negatives = glob.glob("./data/polarLows/processed/negative/*.jpeg")
    combined = positives + negatives
    print(combined)
    random.shuffle(combined)
    sampleCount = len(combined)
    Y = torch.zeros((sampleCount))
    X = torch.zeros((sampleCount, 512, 512, 3))
    i = 0
    for file in combined:
        val = 0
        if "pos" in file:
            val = 1
        Y[i] = val
        img = cv2.imread(file)
        x = torch.from_numpy(img)
        X[i] = x
        i += 1
    torch.save(X, "./data/polarLows/processed/trainingDataX.pt")
    torch.save(Y, "./data/polarLows/processed/trainingDataY.pt")
#At this point assume we have data as tensors
