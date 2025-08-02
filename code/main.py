import splitData
import extractFeatures
import SVM
import visualize

import os


def main():
    print("starting code")
    pathDataset = splitData.main("../data/images", 0.8, 0.1)
    trainFeatures, trainLabels, trainMap, trainAttention = extractFeatures.main(os.path.join(pathDataset, "train"))
    testFeatures, testLabels, testMap, testAttention = extractFeatures.main(os.path.join(pathDataset, "test"))
    validationFeatures, validationLabels, validationMap, validationAttention = extractFeatures.main(os.path.join(pathDataset, "validation"))
    
    if trainMap == testMap == validationMap:
        print("labels for all three sets identical. Continueing...")
    else:
        print("!! labels not identical. Shutting down...")
        return 0
    
    SVM.main(trainFeatures, trainLabels, testFeatures, testLabels)

    visualize.visualizeAttentionMap(validationAttention, "../data/images")

    print("code finished")


if __name__ == "__main__":
    main()