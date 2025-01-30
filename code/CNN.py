import os
import math
import random
import shutil

import torch
import torchvision
import torchvision.transforms as transforms

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

    print("spliting the dataset into: \n", amountTrainImages, " amount of train images \n", amountTestImages, "amount of test images\n", amountImages - amountTrainImages - amountTestImages, "amound of validation images")

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

    os.makedirs(trainF)
    os.makedirs(testF)
    os.makedirs(validationF)

    for image in os.listdir(pathAllImages):
        if image in testImages:
            shutil.copy(pathAllImages + "/" + image, testF)
        if image in trainImages:
            shutil.copy(pathAllImages + "/" + image, trainF)
        if image in validationImages:
            shutil.copy(pathAllImages + "/" + image, validationF)

def main():
    splitData(0.8, 0.1)
    

if __name__ == '__main__':
    main()