import os
import math
import random
import shutil
import time

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

def trainCNN(pTrain, pTest, device):
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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #Load training dataset
    trainSet = ImageFolder(root=trainF, transform=transform)
    trainLoader = DataLoader(trainSet, batch_size=8, shuffle=True, num_workers=6)

    #Load testing dataset
    testSet = ImageFolder(root=testF, transform=transform)
    testLoader = DataLoader(testSet, batch_size=4, shuffle=False, num_workers=6)

    #Load validation dataset
    validationSet = ImageFolder(root=validationF, transform=transform)
    validationLoader = DataLoader(validationSet, batch_size=8, shuffle=False, num_workers=6)


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
            if i == 20:  # print every 2000 mini-batches
                #print('[%d, %5d] loss: %.3f' %
                    #(epoch + 1, i + 1, runningLoss / 20))
                runningLoss = 0.0

    print('Finished Training')

    return model, validationLoader


def evaluateCNN(model, validationLoader, device):
    model.eval()

    correctGuesses = 0
    totalGuesses = 0

    with torch.no_grad():
        for data in validationLoader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  #get predicted class (highest probability)
            
            totalGuesses += labels.size(0)
            correctGuesses += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = 100 * correctGuesses/totalGuesses
    print(f"The accuracy of the model is: {accuracy:.2f}%")
    return accuracy



def main():
    iterations = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Make sure the model uses the GPU if available
    
    accuracyList = []
    totalAccuracy = 0
    startTime = time.time()

    for i in range(iterations):

        model, validationLoader = trainCNN(0.8, 0.1, device)

        accuracy = evaluateCNN(model, validationLoader, device)
        accuracyList.append(accuracy)
        totalAccuracy += accuracy
    
    endTime = time.time()
    duration = endTime - startTime
    durationM, durationS = divmod(duration, 60)

    totalAccuracy = totalAccuracy / iterations
    print(accuracyList)
    print(f"total accuracy is: {totalAccuracy:.2f}%")
    print(f"it took {int(durationM)} minutes and {int(durationS)} seconds to complete the {iterations} iterations")
    
    

if __name__ == '__main__':
    main()