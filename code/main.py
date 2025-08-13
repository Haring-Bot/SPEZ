import splitData
import extractFeatures
import SVM
import visualize
import relevancy
import utils

import os
import numpy as np


def main():
    doSaveModel = False
    doLoadModel = True
    pathImages = "../data/images"
    #pathImages = "../data/animal_images"

    print("starting code")
    
    if doLoadModel:
        loaded_data = utils.loadModel()
        if loaded_data is not None:
            (trainFeatures, trainLabels, trainMap, trainAttention, tokensTrain,
             testFeatures, testLabels, testMap, testAttention, tokensTest,
             validationFeatures, validationLabels, validationMap, 
             validationAttention, tokensValidation) = loaded_data
            
            print("Using loaded model data")
        else:
            doLoadModel = False 
    
    if not doLoadModel:
        pathDataset = splitData.main(pathImages, 0.8, 0.15)
        trainFeatures, trainLabels, trainMap, trainAttention, tokensTrain = extractFeatures.main(os.path.join(pathDataset, "train"))
        testFeatures, testLabels, testMap, testAttention, tokensTest = extractFeatures.main(os.path.join(pathDataset, "test"))
        validationFeatures, validationLabels, validationMap, validationAttention, tokensValidation = extractFeatures.main(os.path.join(pathDataset, "validation"))
        
        if doSaveModel:
            utils.saveModel(trainFeatures, trainLabels, trainMap, trainAttention, tokensTrain,
                           testFeatures, testLabels, testMap, testAttention, tokensTest,
                           validationFeatures, validationLabels, validationMap, 
                           validationAttention, tokensValidation)
    
    if trainMap == testMap == validationMap:
        print("labels for all three sets identical. Continueing...")
    else:
        print("!! labels not identical. Shutting down...")
        return 0
    
    SVMmodel, weights = SVM.main(trainFeatures, trainLabels, testFeatures, testLabels)

    relevancyMap = relevancy.combineAttentionWeight(weights, validationFeatures, validationLabels, validationAttention, tokensValidation)

    # print(f"type attention{type(validationAttention)}")
    # print(f"type relevancyMap{type(relevancyMap)}")

    # # Print shape of first value in both maps
    # print(f"validationAttention first value shape: {next(iter(validationAttention.values())).shape}")
    # print(f"relevancyMap first value shape: {next(iter(relevancyMap.values())).shape}")

    # print(list(validationAttention.keys()))
    # print(list(relevancyMap.keys()))

    # print(f"attention length:{len(validationAttention[next(iter(validationAttention))])}")
    # print(f"relevancy length:{len(relevancyMap[next(iter(validationAttention))])}")

    # print("attention")
    # print(validationAttention[next(iter(validationAttention))])
    # print("relevancy")
    # print(relevancyMap[next(iter(validationAttention))])
    # print(list(validationAttention.keys()))
    # print(list(relevancyMap.keys()))
    visualize.visualizeAttentionMap(validationAttention, pathImages, True)
    visualize.visualizeRelevancyMap(relevancyMap, pathImages, True)

    # Print shape of first key-value pair
    first_key = next(iter(validationAttention))
    print(f"validationAttention['{first_key}'].shape: {validationAttention[first_key].shape}")

    first_key = next(iter(relevancyMap))
    print(f"relevancyMap['{first_key}'].shape: {relevancyMap[first_key].shape}")

    # Uncomment and add this debugging:
    first_key = next(iter(validationAttention))
    attention_data = validationAttention[first_key]

    print(f"Attention statistics for {first_key}:")
    for head_idx in range(attention_data.shape[0]):
        head = attention_data[head_idx]
        print(f"Head {head_idx}: min={head.min():.6f}, max={head.max():.6f}, argmax={head.argmax()}")
        print(f"  Top 5 values: {np.sort(head)[-5:]}")
        print(f"  Top 5 positions: {np.argsort(head)[-5:]}")

    print("code finished")

    

if __name__ == "__main__":
    main()