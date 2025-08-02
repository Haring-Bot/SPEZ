from sklearn.svm import LinearSVC  # MUST change this
from sklearn.metrics import accuracy_score
import numpy as np
import argparse

from config import svmC, svmClassWeight, svmTol, svmRandomSate


def main(featuresTrain, labelsTrain, featuresTest, labelsTest):
    print("training SVM")

    X = featuresTrain
    Y = labelsTrain
    weightDict = {}

    # LinearSVC uses one-vs-rest by default
    model = LinearSVC(
                C = svmC,
                class_weight = svmClassWeight,
                tol = svmTol,
                random_state = svmRandomSate
                ).fit(X, Y)

    pred = model.predict(featuresTest)
    accuracy = accuracy_score(labelsTest, pred)

    print(f"coef shape:{model.coef_.shape[0]}")

    for i in range(model.coef_.shape[0]):
        weights = model.coef_[i]
        #print(f"weights have a length of{len(weights)}")
        weightDict[i] = weights
        

    print(f"The SVM achieves an accuracy of {accuracy*100} %")
    print(pred)

    return model, weightDict



if __name__ == "__main__":
    main()

