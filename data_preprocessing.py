import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data_from_csv(csv_path, base_path):
    data = pd.read_csv(csv_path)
    images = []
    labels = []
    print(f"Reading CSV file: {csv_path}")
    print(f"CSV file content:\n{data.head()}")  # Print first few rows of the CSV for debugging
    
    for index, row in data.iterrows():
        # Construct the image path
        image_path = os.path.join(base_path, row['Filename'])
        print(f"Trying to read image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image could not be read: {image_path}")
            continue
        
        image = cv2.resize(image, (32, 32))
        images.append(image)
        labels.append(row['ClassId'])
    
    return np.array(images), np.array(labels)

# Load training data
train_csv_path = r'C:\Users\Arshi\OneDrive\Desktop\Traffic_Sign_Classification_Project\data\GTSRB\Train\GT-final_train.csv'
train_base_path = r'C:\Users\Arshi\OneDrive\Desktop\Traffic_Sign_Classification_Project\data\GTSRB\Train'
images, labels = load_data_from_csv(train_csv_path, train_base_path)

if images.size == 0:
    print("No images loaded. Please check the CSV file and image paths.")
else:
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )

    datagen.fit(images)

    # Save preprocessed data
    np.save('train_images.npy', images)
    np.save('train_labels.npy', labels)

    print(f'Preprocessed data saved: {len(images)} images and labels')
