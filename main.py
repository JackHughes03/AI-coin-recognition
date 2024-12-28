import pandas as pd
import numpy as np
import os
import cv2
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    RESET = '\033[03m'
    
def load_model():
    print("Model loading")
    
    # Load model
    if not os.path.exists("coin_model.h5"):
        print(f"Error: Model file 'coin_model.h5' not found")
        return
    else:
        model = tf.keras.models.load_model("coin_model.h5")
        print(f"Model loaded successfully")
    
    # Load label mapping
    if not os.path.exists('label_mapping.json'):
        print(f"Error: Label mapping file 'label_mapping.json' not found")
        return
    else:        
        with open('label_mapping.json', 'r') as f:
            label_to_index = json.load(f)
            
            print(f"Map json loaded successfullyf")
            
    print("Passing back to test func ")
        
    return model, label_to_index
        

def test(img_path):
    print("\n" + "="*50)
    print("STARTING COIN DETECTION PROCESS")
    print("="*50 + "\n")
    
    print("LOADING MODEL AND DEPENDENCIES")
    print("-"*30)
    
    # Load the model
    print("Attempting to load model...")
    model, label_to_index = load_model()
    
    if model is None or label_to_index is None:
        print("Error: Model and/or map did not load correctly")
        return
    else:
        print("Model & map loaded successfully")
    
    # Create reverse mapping
    index_to_label = {str(v): k for k, v in label_to_index.items()}
    print("Label mapping created")
    
    print("\nPROCESSING IMAGE")
    print("-"*30)
    print(f"Image path: {img_path}")
    
    IMG_SIZE = 100
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"Error: Image not found at path: {img_path}")
        return
    else:
        print("Image file found")
        
    print("Loading image into memory...")
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        print(f"Error: Failed to load image at {img_path}")
        return
    else:
        print("Image loaded successfully")
    
    print("\nPREPARING IMAGE")
    print("-"*30)    
    print("Resizing image...")
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    print(f"Image resized to {IMG_SIZE}x{IMG_SIZE}")
    
    print("Adding batch dimension...")
    img_array = np.array([img_array])
    print("Batch dimension added")
    
    print("Reshaping image...")
    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    print("Image reshaped for model input")
    
    print("Normalizing pixel values...")
    img_array = img_array / 255.0
    print("Image normalized")
    
    print("\nMAKING PREDICTION")
    print("-"*30)
    print("Running model inference...")
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions)
    predicted_label = index_to_label[str(predicted_class)]
    confidence = float(predictions[0][predicted_class]) * 100
    
    print("Prediction complete!")
    print("\n" + "="*50)
    print(f"RESULT: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*50 + "\n")
    

def train():
    print("Training func called")
    
    print("Categorising data")
    
    dataset_dir = "dataset"
    categories = ["train", "test"]
    name_of_coin_file = "cat_to_name.json"
    IMG_SIZE = 100
    training_data = []
    
    # Check directories and files exist
    if not os.path.exists(dataset_dir):
        print("Dataset directory does not exist")
        return
    
    if not os.path.exists(name_of_coin_file):
        print("Json file does not exist")
        return
        
    # Read coin names
    print("Reading coin names...")
    with open(name_of_coin_file, 'r') as f:
        coin_names = json.load(f)
    
    print("Reading training data...")
    
    # Read images and assign numerical labels directly
    label_to_index = {}  # Dictionary to map coin names to indices
    current_label = 0
    
    for i in range(1, 212):
        path = os.path.join(dataset_dir, categories[0], str(i))
        coin_name = coin_names[str(i)]
        
        # Assign numerical label
        if coin_name not in label_to_index:
            label_to_index[coin_name] = current_label
            current_label += 1
            
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_array, label_to_index[coin_name]])
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
    
    if len(training_data) == 0:
        print("No training data found")
        return
    
    print(f"Data read successfully. Total samples: {len(training_data)}")
    
    # Prepare data for training
    X = np.array([i[0] for i in training_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array([i[1] for i in training_data])
    
    # Normalize pixel values
    X = X / 255.0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Data split successfully")
    
    # Create model
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(label_to_index), activation='softmax')
    ])
    
    print("Model created successfully")
    
    # Compile model
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    print("Model compiled successfully")
    
    # Train model
    history = model.fit(X_train, y_train,
                       epochs=10,
                       validation_data=(X_test, y_test))
    
    print("Model trained successfully")
    
    # Save model with metadata
    model_path = "coin_model.h5"
    model.save(model_path)
    
    # Save label mapping as json
    with open('label_mapping.json', 'w') as f:
        json.dump(label_to_index, f)
    
    print(f"Model saved to {model_path}")
    print("Label mapping saved to label_mapping.json")
    
    
def evaluate():
    print("Evaluation function called")
    


if __name__ == "__main__":
    print("\n******* Project started *******")
    
    import gui
    gui.main()
