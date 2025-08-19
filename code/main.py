import splitData
import extractFeatures
import SVM
import visualize
import relevancy
import utils

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


def main():
    doSaveModel = True
    doLoadModel = True
    pathImages = "../data/images"
    #pathImages = "../data/animal_images"
    saveImages = False
    pathModel = "default"
    nFolds = 5

    print("starting code")
    
    if doLoadModel:
        features, labels, mapping, attentionMap, tokenDict = utils.loadModel(pathModel)
    else:
        pathImagesExtraction = pathImages
        if pathImages == "../data/images": pathImagesExtraction = "../data/images sorted"
        if pathImages == "../data/animal_images": pathImagesExtraction = "../data/animal_images_sorted"
        features, labels, mapping, attentionMap, tokenDict = extractFeatures.main(pathImagesExtraction)
        if doSaveModel:
            utils.saveModel(features, labels, mapping, attentionMap, tokenDict)

    print(f"Features extracted: {len(features)}")
    print(f"Labels extracted: {len(labels)}")
    print(f"AttentionMap keys: {len(attentionMap) if attentionMap else 0}")
    
    if len(features) == 0:
        print("ERROR: No features extracted! Check your image directory.")
        return

    featureArray = np.array(features)
    labelArray = np.array(labels)
    mappingArray = np.array(mapping)
    attentionMapArray = np.array(attentionMap)

    imageNames = list(attentionMap.keys())
    foldAccuracies = []

    # Ensure balanced class distribution across folds
    skfold = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=32)

    allRelevancyMaps = {}

    for fold, (trainIDs, valIDs) in enumerate(skfold.split(featureArray, labelArray)):
        trainFeatures = featureArray[trainIDs].tolist()
        trainLabels = labelArray[trainIDs].tolist()
        valFeatures = featureArray[valIDs].tolist()
        valLabels = labelArray[valIDs].tolist()

        valImages = [imageNames[i] for i in valIDs]

        valAttention = {img: attentionMap[img] for img in valImages}
        valTokens = {img: tokenDict[img] for img in valImages}
    
        accuracy, weights = SVM.main(trainFeatures, trainLabels, valFeatures, valLabels)

        relevancyMap = relevancy.combineAttentionWeight(weights, valFeatures, valLabels, valAttention, valTokens)

        if saveImages:
            visualize.visualizeAttentionMap(valAttention, pathImages, True)
            visualize.visualizeRelevancyMap(relevancyMap, pathImages, True)

        foldAccuracies.append(accuracy)
        allRelevancyMaps |= relevancyMap

    combinedRelevancy = visualize.combineRelevancyMaps(allRelevancyMaps)

    meanAccuracy = np.mean(foldAccuracies)      
    print(f"the mean accuracy is {meanAccuracy * 100}%")

    # # Print shape of first key-value pair
    # first_key = next(iter(validationAttention))
    # print(f"validationAttention['{first_key}'].shape: {validationAttention[first_key].shape}")

    # first_key = next(iter(relevancyMap))
    # print(f"relevancyMap['{first_key}'].shape: {relevancyMap[first_key].shape}")

    # # Uncomment and add this debugging:
    # first_key = next(iter(validationAttention))
    # attention_data = validationAttention[first_key]

    # print(f"Attention statistics for {first_key}:")
    # for head_idx in range(attention_data.shape[0]):
    #     head = attention_data[head_idx]
    #     print(f"Head {head_idx}: min={head.min():.6f}, max={head.max():.6f}, argmax={head.argmax()}")
    #     print(f"  Top 5 values: {np.sort(head)[-5:]}")
    #     print(f"  Top 5 positions: {np.argsort(head)[-5:]}")

    print("code finished")

    

if __name__ == "__main__":
    main()