import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import cm
from skimage.transform import resize

from config import cutoffWeightAttention, transparencyAboveCutoff, cmapType, resultsFolder, classes, relevancyOperations, pathImages, percentile

def boxCoordinates(i, xLim, yLim, rows, cols):
    offsetX = xLim / cols
    offsetY = yLim / rows

    goalX = i % cols * offsetX
    goalY = yLim - ((i // cols) * offsetY)

    return goalX, goalY

def combineFishHeatmap(attentionMap, fishImage):
    background_rgb = fishImage/ 255  # already (H, W, 3)
    alphaBg = np.ones((background_rgb.shape[0], background_rgb.shape[1], 1))
    background = np.dstack([background_rgb, alphaBg])

    attentionMap = resize(attentionMap, (background.shape[0], background.shape[1]), mode='reflect', anti_aliasing=True)

    alphaMask = np.where(attentionMap > cutoffWeightAttention, transparencyAboveCutoff, 0)

    cmap = cm.get_cmap(cmapType)
    overlayRGB = cmap(attentionMap)[:, :, :3]

    overlay = np.dstack([overlayRGB, alphaMask])
    #print(overlay)

    alphaO = overlay[:, :, 3:4]
    alphaBg = background[:, :, 3:4]

    outAlpha = alphaO + alphaBg * (1 - alphaO)

    outRGB = (overlay[:, :, :3] * alphaO + background[:, :, :3] * alphaBg * (1 - alphaO)) / np.clip(outAlpha, 1e-6, 1)

    combined = np.dstack([outRGB, outAlpha])

    combined = np.clip(combined, 0, 1)

    # print("new combined")
    # print(combined)


    return combined

def visualizeAttentionMap(attentionMapDict, saveImages = False):
    xMax = 1400
    yMax = 800
    aspectRatio = yMax / xMax
    
    #print("starting to create attention map")

    def addImage(ax, img, x, y, scale = 1.0, title = "", titleOffset = 50):
        imageBox = OffsetImage(img, zoom = scale)
        ab = AnnotationBbox(imageBox, (x, y), frameon=False)
        plt.text(
            x = x, y = y + titleOffset,
            ha = "center",
            va = "center",
            s = title,
            color = "black",
            fontsize = scale * 20,
            backgroundcolor = "white",
            weight = "bold"
        )
        ax.add_artist(ab)

    def showImage(image, name = "test", doPrint = False):
        print(name)
        if(doPrint):print(image)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(name)
        plt.show()
        
    for imageName, attentionMap in attentionMapDict.items():
        headNo = 0
        attentionImages = {}

        fishImage = mpimg.imread(os.path.join(pathImages, imageName))
        if fishImage.ndim == 2:       #if greyscale
            fishImage = np.stack((fishImage,)*3, axis=-1)

        for head in attentionMap:
            for i in range(len(head)):
                if head[i] < cutoffWeightAttention:
                    head[i] = 0
            singleMap = head.reshape(16, 16)
    
            # print(f"singleMap shape: {singleMap.shape}")
            # print(f"singleMap type: {type(singleMap)}")
    
            # print(f"singleMap min: {singleMap.min():.4f}, max: {singleMap.max():.4f}")
    
            highestPixel = np.max(singleMap)
            max_position = np.unravel_index(np.argmax(singleMap), singleMap.shape)
            #print(f"highestPixel at position {max_position} with a value of {highestPixel}")

            image = combineFishHeatmap(singleMap, fishImage)
            
            attentionImages[headNo] = image
            headNo = headNo + 1

        fig, ax = plt.subplots(figsize=(12, 12*aspectRatio), dpi=100)
        ax.set_facecolor("white")
        ax.set_xlim(0, xMax)
        ax.set_ylim(-100, yMax)

        plt.text(
            x = xMax/2, y = yMax,
            ha = "center",
            va = "center",
            s = imageName,
            color = "black",
            fontsize = 40,
            backgroundcolor = "white",
            weight = "bold"
        )
        addImage(ax, fishImage,200 , yMax/2 - 122, 1.5)

        for i in range(len(attentionImages)):
            x, y = boxCoordinates(i, 900, 830, 3, 4)
            #print(f"x: {x+600}  y: {y}")
            xOffset = x+670
            yOffset = y-275
            #print(f"added image at {xOffset}, {yOffset}")
            addImage(ax, attentionImages[i], xOffset, yOffset, 0.45, f"head {i+1}", titleOffset= -135)
        
        ax.axis("off")

        if saveImages:
            savePath = os.path.join(resultsFolder, "attentionMap", f"{imageName}_attentionMap.png")
            plt.savefig(savePath, dpi = 100, bbox_inches='tight')
            #print(f"image {savePath} was saved")
            plt.close()
        else:
            plt.show()

def visualizeRelevancyMap(relevancyMapDict, saveImages = False):
    #print("starting creating relevancyMap")
    for imageName, relevancyMap in relevancyMapDict.items():
        fishImage = mpimg.imread(os.path.join(pathImages, imageName))

        fig, ax = plt.subplots()
        ax.axis('off')

        ax.imshow(fishImage, interpolation="nearest")

        heatmap = ax.imshow(
            relevancyMap,
            cmap=cmapType,
            alpha=0.5,
            interpolation="nearest"
        )
        plt.colorbar(heatmap, ax=ax)

        # Add title above the image
        ax.text(
            0.5, 1.05,               # x = middle, y = slightly above
            imageName,
            ha="center",
            va="bottom",
            color="black",
            fontsize=20,
            backgroundcolor="white",
            weight="bold",
            transform=ax.transAxes   # use relative axes coordinates
        )
        if saveImages:
            savePath = os.path.join(resultsFolder, "relevancyMap", f"{imageName}_relevancyMap.png")
            plt.savefig(savePath, dpi = 100, bbox_inches='tight')
            #print(f"image {savePath} was saved")
            plt.close()
        else:
            plt.show()

def combineRelevancyMaps(mapDict):
    def saveHeatmap(array, className, operationName):
        fishImagePath = os.path.join(pathImages, f"{className}01.jpg")
        if className == "allClasses":
            fishImagePath = "../data/oreochromis niloticus_modified.png"
        
        savePath = os.path.join(resultsFolder, "summary")
        
        # Filter to show only top X% of relevancies
        array_filtered = array.copy()
        threshold = np.percentile(np.abs(array_filtered[array_filtered != 0]), 100-percentile)
        array_filtered[np.abs(array_filtered) < threshold] = 0  #all else = 0
        
        if os.path.exists(fishImagePath):
            vmax = np.max(np.abs(array_filtered[array_filtered != 0])) if np.any(array_filtered != 0) else 1
            vmin = -vmax
            overlayImage = overlayFishWithRelevancy(fishImagePath, array_filtered, vmin, vmax, 0.5)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(overlayImage)
            ax.axis("off")
            
            cmap = plt.cm.get_cmap(cmapType)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Empty array for the mappable
            
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Relevancy (Top 10%)', rotation=270, labelpad=15)
            
            plt.savefig(os.path.join(savePath, f"{className}_{operationName}_RelevancyMap.png"), 
                       bbox_inches='tight', dpi=100)
            plt.close()
        else:
            # Fallback: regular heatmap without overlay
            plt.figure(figsize=(6, 6))
            vmax = np.max(np.abs(array_filtered[array_filtered != 0])) if np.any(array_filtered != 0) else 1
            vmin = -vmax
            im = plt.imshow(array_filtered, cmap=cmapType, vmin=vmin, vmax=vmax)
            plt.axis("off")
            plt.colorbar(im)
            plt.savefig(os.path.join(savePath, f"{className}_{operationName}_RelevancyMap.png"))
            plt.close()
    
    relevancies = {name: [] for name in classes}
    for indivKey, indivEntry in mapDict.items():
        for indivClass in classes:
            if indivClass in indivKey:
                relevancies[indivClass].append(indivEntry)

    classSummaries = {
        "mean": {name: None for name in classes},
        "median": {name: None for name in classes},
        "std": {name: None for name in classes},
        "max": {name: None for name in classes},
    }
    
    for className in relevancies:
        if relevancyOperations["mean"]:
            mean = np.mean(np.stack(relevancies[className], axis=0), axis=0)
            classSummaries["mean"][className] = mean
            saveHeatmap(mean, className, "mean")
        if relevancyOperations["median"]:
            median = np.median(np.stack(relevancies[className], axis=0), axis=0)
            classSummaries["median"][className] = median
            saveHeatmap(median, className, "median")
        if relevancyOperations["std"]:
            std = np.std(np.stack(relevancies[className], axis=0), axis=0)
            classSummaries["std"][className] = std
            saveHeatmap(std, className, "std")
        if relevancyOperations.get("max", False):
            maxArray = np.max(np.stack(relevancies[className], axis=0), axis=0)
            classSummaries["max"][className] = maxArray
            saveHeatmap(maxArray, className, "max")

    allClasses = []
    for className, relevancyList in relevancies.items():
        allClasses.extend(relevancyList)

    if relevancyOperations.get("mean", False):
        meanAll = np.mean(allClasses, axis=0)
        saveHeatmap(meanAll, "allClasses", "mean")

    if relevancyOperations.get("median", False):
        medianAll = np.median(allClasses, axis=0)
        saveHeatmap(medianAll, "allClasses", "median")

    if relevancyOperations.get("std", False):
        stdAll = np.std(allClasses, axis=0)
        saveHeatmap(stdAll, "allClasses", "std")

    if relevancyOperations.get("max", False):
        maxAll = np.max(allClasses, axis=0)
        saveHeatmap(maxAll, "allClasses", "max")
    
    print(f"Total relevancy maps across all classes: {len(allClasses)}")

    return classSummaries

def overlayFishWithRelevancy(fishImagePath, resultFiltered, vmin, vmax, alpha=0.5):
    fishImage = mpimg.imread(fishImagePath)
    # Handle different image formats properly
    if fishImage.ndim == 2:  # Already grayscale
        fishImage = np.stack([fishImage] * 3, axis=-1)  # Convert to RGB
    elif fishImage.ndim == 3:
        if fishImage.shape[2] == 1:  # Grayscale with extra dimension
            fishImage = np.squeeze(fishImage, axis=2)
            fishImage = np.stack([fishImage] * 3, axis=-1)
        elif fishImage.shape[2] == 4:  # RGBA
            fishImage = fishImage[:, :, :3]
        elif fishImage.shape[2] == 3:  # Already RGB
            pass  # Do nothing
        else:
            print(f"Warning: Unexpected image shape {fishImage.shape}")
            fishImage = fishImage[:, :, :3]

    
    if fishImage.max() > 1: #normalize
        fishImage = fishImage / 255.0
    
    cmap = plt.cm.get_cmap(cmapType)
    
    zero_mask = (resultFiltered == 0)   #zero values remain unchaged
    
    norm_relevancy = (resultFiltered - vmin) / (vmax - vmin)    #normalize relevancy
    norm_relevancy = np.clip(norm_relevancy, 0, 1)
    
    colored_relevancy = cmap(norm_relevancy)[:, :, :3]  #apply relevancy
    
    alpha_mask = np.ones_like(resultFiltered) * alpha
    alpha_mask[zero_mask] = 0  #zero values = transparent
    
    #combine masks
    combined = fishImage * (1 - alpha_mask[:, :, np.newaxis]) + colored_relevancy * alpha_mask[:, :, np.newaxis]
    
    # Ensure values are in valid range
#    combined = np.clip(combined, 0, 1)
    
    return combined

def main():
    print("main")

if __name__ == "__main__":
    main()