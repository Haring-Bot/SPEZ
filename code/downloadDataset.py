import kaggle
import random
import os

# Example: Playing cards dataset
dataset = "likhon148/animal-data"

# Download the dataset to a temp folder
os.makedirs("../alternativeData", exist_ok=True)
kaggle.api.dataset_download_files(dataset, path="alternativeData", unzip=True)

# Keep only ~300 random images
import glob, shutil
imgs = glob.glob("../alternativeData/**/*.jpg", recursive=True)
selected = random.sample(imgs, 300)
os.makedirs("dataset_300", exist_ok=True)
for img in selected:
    shutil.copy(img, "dataset_300")
