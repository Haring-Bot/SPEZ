import os

import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image

def preprocessImage(imagePath):
    image = Image.open(imagePath).convert('RGB')
    return preprocess(image).unsqueeze(0)  #batch dimension for DINO

def grayToRGB(img):
    return img.convert("RGB")

def extractFeatures(imgTensor, model):
    with torch.no_grad():
        features = model(imgTensor)
    return features.squeeze(0).cpu().numpy()

preprocess = transforms.Compose([
    transforms.Lambda(grayToRGB),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stds and mean
])

def main(folderPath = "../data/dataset/train"):
    print("start feature extraction")

    if torch.cuda.is_available():
        print("gpu found. Using cuda.")
        device = torch.device("cuda")
    else:
        print("! No gpu found. Using cpu instead.")
        device = torch.device("cpu")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)

    imageExtensions = (".png", ".jpg", ".jpeg")
    featureSet = []
    labelSet = []
    labelMap = {}

    classFolders = sorted([d for d in os.listdir(folderPath)
                           if os.path.isdir(os.path.join(folderPath, d))])

    for labelIndex, className in enumerate(classFolders):
        labelMap[className] = labelIndex
        classFolderPath = os.path.join(folderPath, className)
        imageFiles = [f for f in os.listdir(classFolderPath)
                      if f.lower().endswith(imageExtensions)
                      and os.path.isfile(os.path.join(classFolderPath, f))]

        for imageFile in imageFiles:
            imagePath = os.path.join(classFolderPath, imageFile)
            imageTensor = preprocessImage(imagePath).to(device)
            features = extractFeatures(imageTensor, model)

            featureSet.append(features)
            labelSet.append(labelIndex)

    print("Finished extracting features.")
    print(labelMap)
    return np.array(featureSet), np.array(labelSet), labelMap

if __name__ == "__main__":
    main()