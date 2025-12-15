# SPEZ: Explainable AI Pipeline for Morphological Classification

This repository details the implementation of a classification model designed not only for high accuracy but also for deep interpretability in the field of morphology.

## Project Summary

* **Goal:** To accurately classify biological specimens (specifically, fish species) based on their body shape while using Explainable Artificial Intelligence (XAI) to identify the exact morphological features driving the classification decision.
* **Pipeline:** We developed a hybrid computer vision pipeline that uses a pre-trained **Vision Transformer (DINOv2)** as a feature extractor, feeding the resulting feature vectors into a linear **Support Vector Machine (SVM)** classifier.
* **High Performance:** The resulting pipeline achieved a high classification accuracy ($\approx 97.6\%$) in categorizing specimens.
* **Key XAI Contribution:** We successfully generated **"Relevancy Maps"** by combining the Vision Transformer's internal attention mechanisms with the SVM's weight vectors.  These maps precisely highlight the morphological regions (landmarks) that are most relevant to the model's prediction.
* **Impact:** Through this explainability technique, we were able to validate known biological features and, crucially, uncover hidden dataset biases (such as the influence of specimen mounting wires, a "Clever-Hans" effect) that would have been invisible in a traditional black-box model.

## Technologies Used

* Python
* PyTorch
* Vision Transformer (DINOv2 architecture)
* Scikit-Learn (for SVM)
* Matplotlib (for Relevancy Maps visualization)
