# ViT-SVM Pipeline for Morphological Classification & XAI

This repository contains the implementation for our research paper focused on applying Explainable Artificial Intelligence (XAI) to the classification of fish species based on their morphology.

    Goal: To build a high-accuracy classification model that also provides deep insight into its decision-making process, helping researchers uncover true morphological landmarks.

    Methodology: We developed a hybrid pipeline combining a pre-trained Vision Transformer (DINOv2) for robust feature extraction with a Support Vector Machine (SVM) classifier.

    Key Achievement: The pipeline achieved a high classification accuracy (≈97.6%) in categorizing fish based on their lake of origin.

    Explainable AI: By combining the Vision Transformer's attention maps with the SVM's weight vectors, we generated "Relevancy Maps". These maps clearly visualize the specific morphological features the model focused on.

    Contribution: This XAI approach successfully identified true biological landmarks while simultaneously detecting and confirming dataset biases, such as the influence of specimen mounting wires ("Clever-Hans" effects).
