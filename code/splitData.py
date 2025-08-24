import os
import math
import random
import shutil

from config import classes, setTypes, pathImages

def main(pTrain = 0.8, pTest = 0.1, skipBalancing=False):
    if pTrain+ pTest > 1:
        print("pTrain and pTest in the splitData function are larger than 1. Please adjust. \n !!Script terminated!!")
        return 0
    elif pTrain + pTest > 0.9:
        print("! Warning, the parameters of the splitData() function leave less then 10% of the dataset for validation use. Please use smalle values!")
    else:
        print("starting to split the data")
    
    #pathImages = "../data/images"


    allTestFolderFull = False
    while allTestFolderFull == False:
        allImages = os.listdir(pathImages)
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
        elif skipBalancing:
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

    for image in os.listdir(pathImages):
        for classType in classes:
            if classType in image:
                imageType = classType

        if image in testImages:
            shutil.copy(pathImages + "/" + image, testF + "/" + imageType)
        if image in trainImages:
            shutil.copy(pathImages + "/" + image, trainF + "/" +  imageType)
        if image in validationImages:
            shutil.copy(pathImages + "/" + image, validationF + "/" +  imageType)

    return os.path.join(datasetF, "train"), os.path.join(datasetF, "validation")
