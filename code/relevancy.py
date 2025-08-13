import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

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

def main():
    print("starting relevancy calculation")

if __name__ == "__main__":
    main()