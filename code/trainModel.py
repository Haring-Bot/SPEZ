import os
import math
import random
import shutil
import time

import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Resize,
)

setTypes = ["train", "test", "validation"]
classes = ["Chamo", "Hawassa", "Koka", "Lan", "Tana", "Ziway"]

def splitDataset(pTrain, pTest):
    if pTrain+ pTest > 1:
        print("pTrain and pTest in the splitData function are larger than 1. Please adjust. \n !!Script terminated!!")
        return 0
    elif pTrain + pTest > 0.9:
        print("! Warning, the parameters of the splitData() function leave less then 10% of the dataset for validation use. Please use smalle values!")
    else:
        print("starting to split the data")
    
    pathAllImages = "../data/images"


    allTestFolderFull = False
    while allTestFolderFull == False:
        allImages = os.listdir(pathAllImages)
        amountImages = len(allImages)
        amountTrainImages = math.floor(amountImages * pTrain)
        amountTestImages = math.floor(amountImages * pTest)

        print("spliting the dataset into: \n", amountTrainImages, " amount of train images \n", amountTestImages, "amount of test images\n", amountImages - amountTrainImages - amountTestImages, "amount of validation images")
        trainImages = []
        for elements in range (amountTrainImages):
            randomChoice = random.choice(allImages)
            trainImages.append(randomChoice)
            allImages.remove(randomChoice)

        testImages = []
        for elements in range (amountTestImages):
            randomChoice = random.choice(allImages)
            testImages.append(randomChoice)
            allImages.remove(randomChoice)

        validationImages = allImages

        if all(any(sub in s for s in testImages) for sub in classes):
            if all(any(sub in s for s in validationImages) for sub in classes):
                allTestFolderFull = True
        else:
            print("first try splitting the images left one folder empty. Trying to split again.")

    # print(len(trainImages))
    # print(len(testImages))
    # print(len(validationImages))

    #copy images in folder
    datasetF = "../data/dataset"
    trainF = "../data/dataset/train"
    testF = "../data/dataset/test"
    validationF = "../data/dataset/validation"

    if os.path.exists(datasetF):
        shutil.rmtree(datasetF)


    for setType in setTypes:
        setTypeP = os.path.join(datasetF, setType)
        os.makedirs(setTypeP)
        for classType in classes:
            os.makedirs(os.path.join(setTypeP, classType))

    for image in os.listdir(pathAllImages):
        for classType in classes:
            if classType in image:
                imageType = classType

        if image in testImages:
            shutil.copy(pathAllImages + "/" + image, testF + "/" + imageType)
        if image in trainImages:
            shutil.copy(pathAllImages + "/" + image, trainF + "/" +  imageType)
        if image in validationImages:
            shutil.copy(pathAllImages + "/" + image, validationF + "/" +  imageType)

    return trainF, testF, validationF

def train(trainF, testF, validationF):
    modelName = "google/vit-large-patch16-224"
    processor = ViTImageProcessor.from_pretrained(modelName)
    processor

    size = processor.size["height"]
    mean, stdDev = processor.image_mean, processor.image_std
    
    trainTransform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size), #does this even make sense since all imgs are te same size?
        torchvision.transforms.RandomHorizontalFlip(),  #sense? All images have the same orientation
        torchvision.transforms.Normalize(mean=mean, std=stdDev),
        torchvision.transforms.ToTensor()
    ])

    testValTransform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size), 
        torchvision.transforms.CenterCrop(size),  #does this even make sense since all imgs are te same size?
        torchvision.transforms.Normalize(mean=mean, std=stdDev),
        torchvision.transforms.ToTensor()
    ])

    #Load training dataset
    trainSet = ImageFolder(root=trainF, transform=trainTransform)
    trainLoader = DataLoader(trainSet, batch_size=8, shuffle=True, num_workers=6)

    #Load testing dataset
    testSet = ImageFolder(root=testF, transform=testValTransform)
    testLoader = DataLoader(testSet, batch_size=4, shuffle=False, num_workers=6)

    #Load validation dataset
    validationSet = ImageFolder(root=validationF, transform=testValTransform)
    validationLoader = DataLoader(validationSet, batch_size=8, shuffle=False, num_workers=6)

    print(trainSet.class_to_idx)
    
    


if __name__ == '__main__':
    print("starting to load the dataset")
    trainFolder, testFolder, validationFolder = splitDataset(0.8, 0.1)
    train(trainFolder, testFolder, validationFolder)
    
