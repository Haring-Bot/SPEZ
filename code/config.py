#general
setTypes = ["train", "test", "validation"]
classes = ["Chamo", "Hawassa", "Koka", "Lan", "Tana", "Ziway"]
#classes = ["Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant", "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"]

#SVM
svmC = 1.0
svmClassWeight = "balanced"
svmTol = 1e-4
svmRandomSate = 42

#visualization
cutoffWeightAttention = 0.01
transparencyAboveCutoff = 0.6
cmapType = "hot"
resultsFolder = "../results"
#resultsFolder = "../results animals"

#analysis
relevancyOperations = {
    "mean" : False,
    "median" : True,
    "std" : False,
    "max" : False,
    "consistency" : False
}