import os
import math
import random
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def splitData(pTrain, pTest):
    if pTrain+ pTest > 1:
        print("pTrain and pTest in the splitData function are larger than 1. Please adjust. \n !!Script terminated!!")
        return 0
    elif pTrain + pTest > 0.9:
        print("! Warning, the parameters of the splitData() function leave less then 10% of the dataset for validation use. Please use smalle values!")
    else:
        print("starting to split the data")
    
    pathAllImages = "../data/images"

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

    setTypes = ["train", "test", "validation"]
    classes = ["Chamo", "Hawassa", "Koka", "Lan", "Tana", "Ziway"]

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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainSet = ImageFolder(root=trainF, transform=transform)
    trainLoader = DataLoader(trainSet, batch_size=8, shuffle=True, num_workers=6)

    # Load testing dataset
    testSet = ImageFolder(root=testF, transform=transform)
    testLoader = DataLoader(testSet, batch_size=4, shuffle=False, num_workers=6)

def main():
    splitData(0.8, 0.1)
    

if __name__ == '__main__':
    main()