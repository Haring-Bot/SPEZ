import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

def combineAttentionWeight(weights, features, labels, attention, tokens):
    for n in range(len(features)):
        #print(n)
        probabilities = []
        imageNames = list(tokens.keys())

        for i in range(len(weights)):
            probabilities.append(np.dot(weights[i], features[n]))
        #     print(f"{i}: {probabilities[i]}")
        prediction= np.argmax(probabilities)
        # print(f"prediction:{np.argmax(probabilities)}")
        # print(f"truth:{labels[n]}")


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

        plt.imshow(uRelevancyMap, cmap='jet')
        plt.title(imageNames[n])
        plt.axis('off')
        plt.show()

def main():
    print("starting relevancy calculation")

if __name__ == "__main__":
    main()