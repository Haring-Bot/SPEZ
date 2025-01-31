import os
import math
import random
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

setTypes = ["train", "test", "validation"]
classes = ["Chamo", "Hawassa", "Koka", "Lan", "Tana", "Ziway"]

def trainCNN(pTrain, pTest):
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

    allTestFolderFull = False
    while allTestFolderFull == False:
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


    #Load preexisting model
    vgg_conv = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg_conv = nn.Sequential(*list(vgg_conv.features.children()))  # Delete all but the conv layers

    for param in list(vgg_conv.parameters())[:-4]:  #All layers except the last 4 can't be trained
        param.requires_grad = False

    class CustomVGG(nn.Module):                         #new classifier
        def __init__(self, vgg_conv, num_classes=6):
            super(CustomVGG, self).__init__()
            self.vgg_conv = vgg_conv                    #Feature extractor
            self.flatten = nn.Flatten()                 #Flatten
            self.fc1 = nn.Linear(512 * 7 * 7, 1024)     #Fully connected layer
            self.relu = nn.ReLU()                       #
            self.dropout = nn.Dropout(0.5)              #Dropout to prevent overfitting
            self.fc2 = nn.Linear(1024, num_classes)     #classifier
            self.softmax = nn.Softmax(dim=1)            #probability for each class

        def forward(self, x):                           #model flow
            x = self.vgg_conv(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x

    model = CustomVGG(vgg_conv, num_classes=6)          #create model

    #Make sure the model uses the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  #Move the model to GPU or CPU

    #print(model)

    lossFunc = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

    print("starting to train the model")

    for epoch in range(70):  # loop over the dataset multiple times

        runningLoss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  #Move inputs and labels to GPU or CPU

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = lossFunc(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, runningLoss / 2000))
                runningLoss = 0.0

    print('Finished Training')

def main():
    trainCNN(0.8, 0.1)
    

if __name__ == '__main__':
    main()