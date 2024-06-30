import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import cv2
import os

# Load Data Function
def load_data(data_dir, csv_file):
    images = []
    labels = []
    try:
        data = pd.read_csv(csv_file)
        for index, row in data.iterrows():
            image_path = os.path.join(data_dir, row['Filename'])
            if os.path.isfile(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (32, 32))
                    images.append(image)
                    labels.append(int(row['ClassId']))
                else:
                    print(f"Error reading image: {image_path}")
            else:
                print(f"File not found: {image_path}")
    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file {csv_file} is empty or not properly formatted.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return np.array(images), np.array(labels)

# Paths
test_data_dir = r'C:\Users\Arshi\OneDrive\Desktop\Traffic_Sign_Classification_Project\data\GTSRB\Test'
test_csv_path = r'C:\Users\Arshi\OneDrive\Desktop\Traffic_Sign_Classification_Project\data\GTSRB\Test\GT-final_test.csv'

# Load data
test_images, test_labels = load_data(test_data_dir, test_csv_path)

if test_images.size == 0 or test_labels.size == 0:
    print("No test data loaded. Please check the CSV file and the test data directory.")
else:
    test_images = test_images / 255.0  # Normalize images
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=43)  # One-hot encode labels

    print(f'Test images shape: {test_images.shape}')
    print(f'Test images type: {type(test_images)}')
    print(f'Test labels shape: {test_labels.shape}')
    print(f'Test labels type: {type(test_labels)}')

    try:
        # Load the model
        model_path = 'traffic_sign_classifier.h5'
        if os.path.isfile(model_path):
            model = load_model(model_path)
            model.summary()  # Print model summary for debugging

            # Evaluate the model
            loss, accuracy = model.evaluate(test_images, test_labels)
            print(f'Test accuracy: {accuracy}')
        else:
            print(f"Model file not found: {model_path}")
    except Exception as e:
        print(f"An error occurred while loading or evaluating the model: {e}")
