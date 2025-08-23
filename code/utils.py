import os
import pickle
import re
from datetime import datetime

def saveModel(featuresT, labelsT, mappingT, attentionMapT, tokenDictT, featuresV, labelsV, mappingV, attentionMapV, tokenDictV):
    date_str = datetime.now().strftime("%Y%m%d%H%M")
    saveFile = f"../models/model_{date_str}.pkl"
    saveData = {
        'featuresT': featuresT,
        'labelsT': labelsT,
        'mappingT': mappingT,
        'attentionMapT': attentionMapT,
        'tokenDictT': tokenDictT,
        'featuresV': featuresV,
        'labelsV': labelsV,
        'mappingV': mappingV,
        'attentionMapV': attentionMapV,
        'tokenDictV': tokenDictV
    }
    
    # Create models directory if it doesn't exist
    os.makedirs("../models", exist_ok=True)
    
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
        
        path = os.path.join(folderPath, newestFile)

    if os.path.exists(path):
        print(f"loading model from {path}...")
        with open(path, "rb") as f:
            savedData = pickle.load(f)

        # Extract all the data from saved dictionary
        featuresT = savedData['featuresT']
        labelsT = savedData['labelsT']
        mappingT = savedData['mappingT']
        attentionMapT = savedData['attentionMapT']
        tokenDictT = savedData['tokenDictT']
        featuresV = savedData['featuresV']
        labelsV = savedData['labelsV']
        mappingV = savedData['mappingV']
        attentionMapV = savedData['attentionMapV']
        tokenDictV = savedData['tokenDictV']
        
        print("Model data loaded successfully!")
        
        # Return all variables in the same order as main.py expects
        return (featuresT, labelsT, mappingT, attentionMapT, tokenDictT, 
                featuresV, labelsV, mappingV, attentionMapV, tokenDictV)
    else:
        print(f"path {path} couldn't be found. Starting new training...")
        return None