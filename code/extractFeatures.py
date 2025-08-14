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

def extractAttention(model, imageTensor):
    attentionScores = []

    # Save original forward method
    attentionModule = model.blocks[-1].attn
    originalForward = attentionModule.forward

    def forward_hooked(x):
        qkv = attentionModule.qkv(x)
        B, N, C3 = qkv.shape
        C = C3 // 3

        num_heads = 12  
        head_dim = C // num_heads

        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)

        attentionScores.append(attn.detach().cpu())

        out = (attn @ v).transpose(2, 3).reshape(B, N, C)
        out = attentionModule.proj(out)
        out = attentionModule.proj_drop(out)

        return out

    attentionModule.forward = forward_hooked

    with torch.no_grad():
        _ = model(imageTensor)

    attentionModule.forward = originalForward

    return attentionScores[0] if attentionScores else None

def main(folderPath = "../data/images"):
    print("start feature extraction")

    if torch.cuda.is_available():
        print("gpu found. Using cuda.")
        device = torch.device("cuda")
    else:
        print("! No gpu found. Using cpu instead.")
        device = torch.device("cpu")

    #model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    #model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True, force_reload=True)
    model.eval().to(device)

    imageExtensions = (".png", ".jpg", ".jpeg")
    featureSet = []
    labelSet = []
    labelMap = {}
    attentionMaps = {}
    featuresToImages = {}
    tokenDict = {}

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

            # # Debugging: Print image tensor shape and stats
            # print(f"Image tensor shape: {imageTensor.shape}")
            # print(f"Image tensor stats: min={imageTensor.min()}, max={imageTensor.max()}, mean={imageTensor.mean()}")

            # # Save a processed image to check visually
            # import torchvision.transforms as T
            # T.ToPILImage()(imageTensor.squeeze(0)).save("debug_processed.png")

            with torch.no_grad():
                features = model(imageTensor)
                attention = extractAttention(model, imageTensor)
                tokens = model.forward_features(imageTensor)
                tokenDict[imageFile] = tokens["x_norm_patchtokens"].squeeze(0).cpu().numpy()

            # Fix: Skip CLS token (index 0) AND register tokens (indices 1-4)
            # Only take patch tokens (indices 5 onwards)
            clsAttention = attention[0, :, 0, 5:]  # Changed from 1: to 5:]
            
            #print(f"Raw attention shape: {attention.shape}")
            #print(f"Raw attention sample: {attention[0, 0, :5, :5]}")

            # print(f"length features{len(features)}")
            # print(features.shape)
            # print(f"length attention{len(clsAttention)}")
            # print(clsAttention.shape)
            featureSet.append(features.squeeze(0).cpu().numpy())
            labelSet.append(labelIndex)
            featuresToImages[len(featureSet)] = imageFile
            attentionMaps[imageFile] = clsAttention.cpu().numpy()

    print("Finished extracting features.")
    
    # attention = model.get_last_selfattention(imageTensor)
    # clsAttention = attention[0, :, 0, 1:]

    return np.array(featureSet), np.array(labelSet), labelMap, attentionMaps, tokenDict

if __name__ == "__main__":
    main()