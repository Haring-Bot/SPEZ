import os
import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import image as mpimg
import visualize

from config import relevancyOperations, cmapType, resultsFolder, topPercentile, lowPercentile, pathImages, sampleImagePath

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
        
        if prediction == truth:
            correctPredictions += 1
        totalPredictions += 1
        

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
    
    accuracy = correctPredictions / totalPredictions
    print(f"Validation Accuracy: {accuracy:.4f} ({correctPredictions}/{totalPredictions})")
    
    return relevancyMaps

def relevancySubstractions(classRelevancies):
    
    fig = plt.figure(figsize=(17, 18))
    
    gs = gridspec.GridSpec(8, 8, figure=fig, 
                          height_ratios=[0.3, 0.3, 1, 1, 1, 1, 1, 1],  # Added extra row at top
                          width_ratios=[0.5, 1, 1, 1, 1, 1, 1, 0.25],  # Added extra column at left
                          hspace=0.05,  #vertical spacing
                          wspace=0.05)  #horizontal spacing
    
    title_ax = fig.add_subplot(gs[0, 1:7])  # Adjusted for new grid
    title_ax.text(0.5, 0.5, 'Subtractive Relevancy Maps', 
                  ha='center', va='center', fontsize=30, weight='bold')
    title_ax.axis('off')
    
    # Y-axis label on the LEFT side
    y_label_ax = fig.add_subplot(gs[2:8, 0])  # Left column, spanning main grid rows
    y_label_ax.text(0.5, 0.5, 'Reference Classes A', 
                    ha='center', va='center', fontsize=18, weight='bold', 
                    rotation=90)
    y_label_ax.axis('off')
    
    # X-axis label at the TOP RIGHT
    x_label_ax = fig.add_subplot(gs[1, 1:7])  # Top row, spanning main grid columns
    x_label_ax.text(0.5, 0.5, 'Subtracted Classes B', 
                    ha='center', va='center', fontsize=18, weight='bold')
    x_label_ax.axis('off')
    
    axes = np.empty((6, 6), dtype=object)
    for i in range(6):
        for j in range(6):
            axes[i, j] = fig.add_subplot(gs[i+2, j+1])  # Adjusted for new grid layout
    
    #for colorbar
    global_vmin = float('inf')
    global_vmax = float('-inf')
    
    for operation, isEnabled in relevancyOperations.items():
        nClass1 = 0
        nClass2 = 0
        if isEnabled and operation in classRelevancies:
            allClasses = list(classRelevancies[operation].keys())
            
            for class1 in classRelevancies[operation]:
                for class2 in classRelevancies[operation]:
                    if class1 == class2:# One vs Rest case: current class vs average of all others
                        otherClasses = [cls for cls in allClasses if cls != class1]
                        if otherClasses:
                            othersAvg = np.mean([classRelevancies[operation][cls] for cls in otherClasses], axis=0)
                            result = classRelevancies[operation][class1] - othersAvg
                    else:
                        result = classRelevancies[operation][class1] - classRelevancies[operation][class2]

                    #top bottom percent filter
                    resultFiltered = applyPercentileMask(result)
                    
                    # Use symmetric normalization to ensure 0 maps to white
                    vmax = np.max(np.abs(resultFiltered[resultFiltered != 0]))  # Exclude the 0 values
                    vmin = -vmax
                    
                    # Track global min/max for colorbar
                    global_vmin = min(global_vmin, vmin)
                    global_vmax = max(global_vmax, vmax)

                    fishImagePath = sampleImagePath
                    combinedImage = visualize.overlayFishWithRelevancy(fishImagePath, resultFiltered, vmin, vmax, 0.5)
                    
                    # Display the overlay image
                    axes[nClass1, nClass2].imshow(combinedImage)
                    axes[nClass1, nClass2].set_xticks([])
                    axes[nClass1, nClass2].set_yticks([])

                    if nClass1 == 0:
                        axes[nClass1, nClass2].set_title(class2, fontsize=20)
                    if nClass2 == 0:
                        axes[nClass1, nClass2].set_ylabel(class1, fontsize=20, rotation=90)

                    nClass2 += 1
                nClass1 += 1
                nClass2 = 0
    
    # Create proper colorbar using ScalarMappable
    cbar_ax = fig.add_subplot(gs[2:8, 7])  # Adjusted for new grid
    cmap = plt.cm.get_cmap(cmapType)
    norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Relevancy Difference Top{topPercentile}%", rotation=270, labelpad=15)
    
    plt.tight_layout()
    savePath = os.path.join(resultsFolder, "summary", "subtractive_relevancyMap.png")
    plt.savefig(savePath, dpi=100, bbox_inches='tight')
    #plt.show()

def applyPercentileMask(image):
    topThreshold = np.percentile(image, 100 - topPercentile)
    bottomThreshold = np.percentile(image, lowPercentile)

    mask = (image >= topThreshold) | (image <= bottomThreshold) #create combined mask

    result = image.copy()
    result[~mask] = 0  #all else = 0
    
    return result

# def main():
#     print("starting relevancy calculation")

# if __name__ == "__main__":
#     main()