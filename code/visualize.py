import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import cm
from skimage.transform import resize

from config import cutoffWeightAttention, transparencyAboveCutoff, cmapType

def boxCoordinates(i, xLim, yLim, rows, cols):
    offsetX = xLim / cols
    offsetY = yLim / rows

    goalX = i % cols * offsetX
    goalY = yLim - ((i // cols) * offsetY)

    return goalX, goalY

def visualizeAttentionMap(attentionMapDict, pathImages):
    xMax = 1169
    yMax = 827
    
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

    def combineFishHeatmap(attentionMap):
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

            image = combineFishHeatmap(singleMap)
            
            attentionImages[headNo] = image
            headNo = headNo + 1

        fig, ax = plt.subplots(figsize=(xMax/10, yMax/10), dpi=100)
        ax.set_facecolor("white")
        ax.set_xlim(0, xMax)
        ax.set_ylim(0, yMax)

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
        addImage(ax, fishImage,200 , yMax/2 - 100, 2.5)

        for i in range(len(attentionImages)):
            x, y = boxCoordinates(i, 700, 830, 3, 4)
            #print(f"x: {x+600}  y: {y}")
            xOffset = x+630
            yOffset = y-225
            addImage(ax, attentionImages[i], xOffset, yOffset, 0.70, f"head {i+1}", titleOffset= -130)
        
        ax.axis("off")
        plt.show()


def main():
    print("main")

if __name__ == "__main__":
    main()