import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model
model = load_model('traffic_sign_classifier.h5')

# Classify Image Function
def classify_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image path does not exist or could not be read: {image_path}")
        return None, None
    image = cv2.resize(image, (32, 32))
    image = image / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_id = np.argmax(prediction)
    return class_id, prediction

# Test the function
image_path = r'C:\Users\Arshi\OneDrive\Desktop\Traffic_Sign_Classification_Project\data\GTSRB\Test\00983.png'  # Replace with the path to your test image
class_id, prediction = classify_image(image_path)
if class_id is not None:
    labels = ["Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)", 
              "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)", 
              "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons", "Right-of-way at the next intersection", 
              "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry", 
              "General caution", "Dangerous curve to the left", "Dangerous curve to the right", "Double curve", 
              "Bumpy road", "Slippery road", "Road narrows on the right", "Road work", "Traffic signals", "Pedestrians", 
              "Children crossing", "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits", 
              "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", 
              "Keep right", "Keep left", "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons"]

    label = labels[class_id]
    print(f'The predicted class ID for the image is: {class_id} ({label})')

    # Display the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f'Predicted: {label} (ID: {class_id})')
    plt.show()
else:
    print("Classification failed.")
