import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load Data
images = np.load('train_images.npy')
labels = np.load('train_labels.npy')

# Debug: Print shapes of loaded data
print(f"Loaded images shape: {images.shape}")
print(f"Loaded labels shape: {labels.shape}")

# Normalize images
images = images / 255.0

# One-hot encode labels
labels = tf.keras.utils.to_categorical(labels, num_classes=43)

# Debug: Print shapes after normalization and one-hot encoding
print(f"Images shape after normalization: {images.shape}")
print(f"Labels shape after one-hot encoding: {labels.shape}")

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Debug: Print shapes after train-validation split
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
)

# Simplified Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')  # 43 classes for the GTSRB dataset
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary to debug shapes
model.summary()

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                    epochs=30, 
                    validation_data=(X_val, y_val))

# Save the model
model.save('traffic_sign_classifier.h5')
