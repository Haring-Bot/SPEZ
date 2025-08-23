import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

from config import relevancyOperations, cmapType

def combineAttentionWeight(weights, features, labels, attention, tokens):
    relevancyMaps = {}
    correctPredictions = 0
    totalPredictions = 0
    
    for n in range(len(features)):
        probabilities = []
        imageNames = list(tokens.keys())

        for i in range(len(weights)):
            probabilities.append(np.dot(weights[i], features[n]))
        
        prediction = np.argmax(probabilities)
        truth = labels[n]
        
        # Count correct predictions
        if prediction == truth:
            correctPredictions += 1
        totalPredictions += 1
        
        # Optional: Print individual predictions
        # print(f"Image {n}: Prediction={prediction}, Truth={truth}, Correct={prediction==truth}")

        relevance = np.dot(tokens[imageNames[n]], weights[prediction])
        attentionImage = attention[imageNames[n]].reshape(12, -1)
        averageAttention = attentionImage.mean(axis=0)
        combinedRelevance = relevance * averageAttention
        grid = int(np.sqrt(len(combinedRelevance)))
        relevanceMap = combinedRelevance.reshape(grid, grid)

        relevanceTensor = torch.tensor(relevanceMap).unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(relevanceTensor, size=(224, 224), mode="bilinear", align_corners=False)
        uRelevancyMap = upsampled.squeeze().numpy()

        uRelevancyMap -= uRelevancyMap.min()
        uRelevancyMap /= uRelevancyMap.max()

        relevancyMaps[imageNames[n]] = uRelevancyMap
    
    # Calculate and print accuracy
    accuracy = correctPredictions / totalPredictions
    print(f"Validation Accuracy: {accuracy:.4f} ({correctPredictions}/{totalPredictions})")
    
    return relevancyMaps

def relevancySubstractions(classRelevancies):
    rows, cols = 6, 6
    fig, axes = plt.subplots(rows, cols, figsize = (12,12,))
    
    
    for operation, isEnabled in relevancyOperations.items():
        nClass1 = 0
        nClass2 = 0
        if isEnabled and operation in classRelevancies:
            for class1 in classRelevancies[operation]:
                for class2 in classRelevancies[operation]:
                    print(f"subtracting {class2} from {class1}")
                    if class1 == class2:
                        result = classRelevancies[operation][class1]
                    else:
                        result = classRelevancies[operation][class1] - classRelevancies[operation][class2]
                    posImg = nClass1 * cols + nClass2

                    axes[nClass1, nClass2].imshow(result, cmap=cmapType)
                    axes[nClass1, nClass2].set_xticks([])
                    axes[nClass1, nClass2].set_yticks([])

                    if nClass1 == 0:
                        axes[nClass1, nClass2].set_title(class2, fontsize=20)
                    if nClass2 == 0:
                        axes[nClass1, nClass2].set_ylabel(class1, fontsize=20, rotation=90)


                    nClass2 += 1
                nClass1 += 1
                nClass2 = 0

    plt.tight_layout()
    plt.show()



# def main():
#     print("starting relevancy calculation")

# if __name__ == "__main__":
#     main()