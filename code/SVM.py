from sklearn.svm import LinearSVC  # MUST change this
from sklearn.metrics import accuracy_score
import numpy as np
import argparse

from config import svmC, svmClassWeight, svmTol, svmRandomSate


def main(featuresTrain, labelsTrain, featuresTest, labelsTest):
    #print("training SVM")

    X = featuresTrain
    Y = labelsTrain
    weightDict = {}
    errors = []

    model = LinearSVC(
                C = svmC,
                class_weight = svmClassWeight,
                tol = svmTol,
                random_state = svmRandomSate
                ).fit(X, Y)

    pred = model.predict(featuresTest)
    accuracy = accuracy_score(labelsTest, pred)
    for i in range(len(pred)):
        if pred[i] != labelsTest[i]:
            #print(f"Sample {i}: label={labelsTest[i]}, pred={pred[i]}")
            errors.append(f"{labelsTest[i]}/{pred[i]}")

    for i in range(model.coef_.shape[0]):
        weights = model.coef_[i]
        #print(f"weights have a length of{len(weights)}")
        weightDict[i] = weights
        

    print(f"The SVM achieves an accuracy of {accuracy*100} %")
    #print(pred)

    return accuracy, weightDict, errors



if __name__ == "__main__":
    main()

