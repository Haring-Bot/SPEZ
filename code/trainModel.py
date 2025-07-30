import os
import math
import random
import shutil
import time

import numpy as np
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Resize,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay   
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

    return datasetF

def train(datasetF, device):
    #modelName = "google/vit-large-patch16-224"
    modelName = "google/vit-base-patch16-224"
    

    dataset = load_dataset("imagefolder", data_dir = datasetF)

    processor = ViTImageProcessor.from_pretrained(modelName)
    size = processor.size["height"]
    mean = processor.image_mean
    std = processor.image_std
    
    trainTransforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    valTransforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    
    def applyTrainTransforms(examples):
        examples["pixel_values"] = [trainTransforms(image.convert("RGB")) for image in examples["image"]]
        return examples

    def applyValTransforms(examples):
        examples["pixel_values"] = [valTransforms(image.convert("RGB")) for image in examples["image"]]
        return examples    
    
    id2label = {id: label for id, label in enumerate(dataset["train"].features["label"].names)}
    label2id = {label: id for id, label in id2label.items()}

    dataset["train"].set_transform(applyTrainTransforms)
    dataset["test"].set_transform(applyValTransforms)
    dataset["validation"].set_transform(applyValTransforms)

    def collate_fn(batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        labels = torch.tensor([example["label"] for example in batch])
        return{"pixel_values": pixel_values, "labels": labels}

    labels  = dataset['train'].features['label'].names
    print(labels)


    model = ViTForImageClassification.from_pretrained(
        modelName, 
        num_labels = len(labels),
        id2label=id2label, 
        label2id=label2id, 
        ignore_mismatched_sizes=True
    )

    model.to(device)

    train_args = TrainingArguments(
        output_dir="output-models",
        per_device_train_batch_size=4,
        eval_strategy="epoch",
        num_train_epochs=15,
        fp16=False,
        save_strategy="epoch",
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
        tokenizer=processor,
    )
    trainer.train()

    outputs = trainer.predict(dataset["test"])
    print(outputs.metrics)

    target_names = id2label.values()

    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    labels = dataset["train"].features["label"].names
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)

    print(classification_report(y_true, y_pred, target_names=target_names))

if __name__ == '__main__':
    print("starting to split the dataset")
    datasetFolder = splitDataset(0.8, 0.1)
    print("starting to train the model")
    if torch.cuda.is_available():
        print("gpu found. Using cuda.")
        device = torch.device("cuda")
    else:
        print("No gpu found. Using cpu instead.")
        device = torch.device("cpu")

    train(datasetFolder, device)

    print("code finished")

