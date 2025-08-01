from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import argparse

from config import svmC, svmClassWeight, svmTol, svmRandomSate


def main(featuresTrain, labelsTrain, featuresTest, labelsTest):
    print("training SVM")

    X = featuresTrain
    Y = labelsTrain

    model = SVC(kernel="linear",
                C = svmC,
                class_weight = svmClassWeight,
                tol = svmTol,
                random_state = svmRandomSate
                ).fit(X, Y)

    pred = model.predict(featuresTest)
    accuracy = accuracy_score(labelsTest, pred)

    print(f"The SVM achieves an accuracy of {accuracy*100} %")
    print(pred)



if __name__ == "__main__":
    main()

