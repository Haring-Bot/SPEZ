import splitData
import extractFeatures
import SVM
import visualize
import relevancy

import os


def main():
    print("starting code")
    pathDataset = splitData.main("../data/images", 0.8, 0.1)
    trainFeatures, trainLabels, trainMap, trainAttention, tokensTrain = extractFeatures.main(os.path.join(pathDataset, "train"))
    testFeatures, testLabels, testMap, testAttention, tokensTest = extractFeatures.main(os.path.join(pathDataset, "test"))
    validationFeatures, validationLabels, validationMap, validationAttention, tokensValidation = extractFeatures.main(os.path.join(pathDataset, "validation"))
    
    if trainMap == testMap == validationMap:
        print("labels for all three sets identical. Continueing...")
    else:
        print("!! labels not identical. Shutting down...")
        return 0
    
    SVMmodel, weights = SVM.main(trainFeatures, trainLabels, testFeatures, testLabels)

    relevancy.combineAttentionWeight(weights, validationFeatures, validationLabels, validationAttention, tokensValidation)

    visualize.visualizeAttentionMap(validationAttention, "../data/images")



    print("code finished")


if __name__ == "__main__":
    main()