import os
import pickle
import re
from datetime import datetime

def saveModel(features, labels, mapping, attentionMap, tokenDict):
    date_str = datetime.now().strftime("%Y%m%d%H%M")
    saveFile = f"../models/model_{date_str}.pkl"
    saveData = {
        'features': features,
        'labels': labels,
        'mapping': mapping,
        'attentionMap': attentionMap,
        "tokenDict" : tokenDict
    }
    with open(saveFile, "wb") as f:
        pickle.dump(saveData, f)

    print(f"data saved successfully to {saveFile}")
    return True

def loadModel(path="default"):
    if path == "default":
        folderPath = "../models"
        if not os.path.exists(folderPath):
            print(f"Models folder {folderPath} doesn't exist")
            return None
            
        pattern = re.compile(r"(\d{12})")  # Matches yyyymmddhhmm
        newestFile = None
        newestDate = None

        for fname in os.listdir(folderPath):
            match = pattern.search(fname)
            if match:
                date_str = match.group(1)
                try:
                    date_val = datetime.strptime(date_str, "%Y%m%d%H%M")
                    if newestDate is None or date_val > newestDate:
                        newestDate = date_val
                        newestFile = fname
                except ValueError:
                    pass
        
        if newestFile is None:
            print("No valid model files found")
            return None
        
        path = os.path.join(folderPath, newestFile)  # Fix: join with folder path

    if os.path.exists(path):
        print(f"loading model from {path}...")
        with open(path, "rb") as f:
            savedData = pickle.load(f)

        # Extract the data from saved dictionary
        features = savedData['features']
        labels = savedData['labels']
        mapping = savedData['mapping']
        attentionMap = savedData['attentionMap']
        tokenDict = savedData["tokenDict"]
        
        print("Features loaded successfully!")
        
        # Return all the variables as a tuple
        return (features, labels, mapping, attentionMap, tokenDict)
    else:
        print(f"path {path} couldn't be found. Starting new training...")
        return None