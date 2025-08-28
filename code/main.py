import splitData
import extractFeatures
import SVM
import visualize
import relevancy
import utils

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import pathImages


def main():
    doSaveModel = False
    doLoadModel = True
    saveImages = True
    pathModel = "default"
    nFolds = 5

    print("starting code")
    
    if doLoadModel:
        featuresT, labelsT, mappingT, attentionMapT, tokenDictT, featuresV, labelsV, mappingV, attentionMapV, tokenDictV = utils.loadModel(pathModel)
    else:
        pathTrain, pathValidation = splitData.main(pTrain=0.8, pTest=0, skipBalancing=True)
        pathImagesExtraction = pathImages
        #if pathImages == "../data/images": pathImagesExtraction = "../data/images sorted"
        #if pathImages == "../data/animal_images": pathImagesExtraction = "../data/animal_images_sorted"
        featuresT, labelsT, mappingT, attentionMapT, tokenDictT = extractFeatures.main(pathTrain)
        featuresV, labelsV, mappingV, attentionMapV, tokenDictV = extractFeatures.main(pathValidation)
        if doSaveModel:
            utils.saveModel(featuresT, labelsT, mappingT, attentionMapT, tokenDictT, featuresV, labelsV, mappingV, attentionMapV, tokenDictV)

    print(f"Features extracted: {len(featuresT)}")
    print(f"Labels extracted: {len(labelsT)}")
    print(f"AttentionMap keys: {len(attentionMapT) if attentionMapT else 0}")
    
    if len(featuresT) == 0:
        print("ERROR: No features extracted! Check your image directory.")
        return

    featureArray = np.array(featuresT)
    labelArray = np.array(labelsT)
    mappingArray = np.array(mappingT)
    attentionMapArray = np.array(attentionMapT)

    # Use validation data for cross-validation
    featureArrayV = np.array(featuresV)
    labelArrayV = np.array(labelsV)
    imageNames = list(attentionMapV.keys())
    foldAccuracies = []

    averageAttention = visualize.averageAttentionMap(attentionMapV)
    visualize.visualizeAttentionMap(averageAttention, True)

    # Ensure balanced class distribution across folds
    skfold = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=32)

    allRelevancyMaps = {}
    totalErrors = []

    for fold, (trainIDs, valIDs) in enumerate(skfold.split(featureArrayV, labelArrayV)):
        trainFeatures = featureArrayV[trainIDs].tolist()
        trainLabels = labelArrayV[trainIDs].tolist()
        valFeatures = featureArrayV[valIDs].tolist()
        valLabels = labelArrayV[valIDs].tolist()

        valImages = [imageNames[i] for i in valIDs]

        valAttention = {img: attentionMapV[img] for img in valImages}
        valTokens = {img: tokenDictV[img] for img in valImages}
    
        accuracy, weights, errors = SVM.main(trainFeatures, trainLabels, valFeatures, valLabels)
        for error in errors:
            totalErrors.append(error)

        relevancyMap = relevancy.combineAttentionWeight(weights, valFeatures, valLabels, valAttention, valTokens)

        if saveImages:
            visualize.visualizeAttentionMap(valAttention,True)
            visualize.visualizeRelevancyMap(relevancyMap,True)

        foldAccuracies.append(accuracy)
        allRelevancyMaps |= relevancyMap

    classRelevancies = visualize.combineRelevancyMaps(allRelevancyMaps)

    relevancy.relevancySubstractions(classRelevancies)

    meanAccuracy = np.mean(foldAccuracies)      
    print(f"the mean accuracy is {meanAccuracy * 100}%")
    print("Errors were: pred/ truth")
    print(totalErrors)

    print("code finished")

    

if __name__ == "__main__":
    main()