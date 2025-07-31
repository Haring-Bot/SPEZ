import numpy as np
import os
from sklearn.model_selection import train_test_split  # Import train_test_split from sklearn.model_selection
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical


def load_images(path):
    images = []
    labels = []

    for label in os.listdir(path):
        label_folder = os.path.join(path, label)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            img = Image.open(img_path)
            images.append(np.array(img))
            labels.append(int(label)-1)
    return np.array(images), np.array(labels)

images, labels = load_images("/home/julian/Documents/Spezialisierung/NT_BodyParts/Data/images sorted")
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

print("starting to train")

numFilters = 8
filterSize = 3
poolSize = 3

model = Sequential([
    Conv2D(numFilters, filterSize, input_shape = (224, 224, 1)),
    MaxPooling2D(pool_size=poolSize),
    Flatten(),
    Dense(6, activation="softmax"),
    ])

model.compile(
    "adam", 
    loss = "categorical_crossentropy", 
    metrics=["accuracy"]
    )

model.fit(
    train_images,
    to_categorical(train_labels),
    epochs = 3,
    validation_data=(test_images, to_categorical(test_labels))
)
print(np.unique(labels))