import pandas as pd
import numpy as np
import os
import cv2
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import threading
from colorama import init, Fore, Style
init(autoreset=True)  # Initialize colorama

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    RESET = '\033[03m'
    
def load_model():
    print(f"{Fore.CYAN}Model loading{Style.RESET_ALL}")
    
    # Load model
    if not os.path.exists("coin_model.h5"):
        print(f"{Fore.RED}Error: Model file 'coin_model.h5' not found{Style.RESET_ALL}")
        return
    else:
        model = tf.keras.models.load_model("coin_model.h5")
        print(f"{Fore.GREEN}Model loaded successfully{Style.RESET_ALL}")
    
    # Load label mapping
    if not os.path.exists('label_mapping.json'):
        print(f"{Fore.RED}Error: Label mapping file 'label_mapping.json' not found{Style.RESET_ALL}")
        return
    else:        
        with open('label_mapping.json', 'r') as f:
            label_to_index = json.load(f)
            
            print(f"{Fore.GREEN}Map json loaded successfully{Style.RESET_ALL}")
            
    print(f"{Fore.CYAN}Passing back to test func{Style.RESET_ALL}")
        
    return model, label_to_index
        

def test():
    print("\n" + "="*50)
    print(f"{Fore.CYAN}STARTING COIN DETECTION PROCESS{Style.RESET_ALL}")
    print("="*50 + "\n")
    
    print(f"{Fore.CYAN}LOADING MODEL AND DEPENDENCIES{Style.RESET_ALL}")
    print("-"*30)
    
    img_path = input(f"{Fore.YELLOW}Enter the path to the image file: {Style.RESET_ALL}")
    
    # Load the model
    print(f"{Fore.CYAN}Attempting to load model...{Style.RESET_ALL}")
    model, label_to_index = load_model()
    
    if model is None or label_to_index is None:
        print(f"{Fore.RED}Error: Model and/or map did not load correctly{Style.RESET_ALL}")
        return
    else:
        print(f"{Fore.GREEN}Model & map loaded successfully{Style.RESET_ALL}")
    
    # Create reverse mapping
    index_to_label = {str(v): k for k, v in label_to_index.items()}
    print(f"{Fore.GREEN}Label mapping created{Style.RESET_ALL}")
    
    print("\nPROCESSING IMAGE")
    print("-"*30)
    print(f"Image path: {img_path}")
    
    IMG_SIZE = 100
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"{Fore.RED}Error: Image not found at path: {img_path}{Style.RESET_ALL}")
        return
    else:
        print(f"{Fore.GREEN}Image file found{Style.RESET_ALL}")
        
    print("Loading image into memory...")
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        print(f"{Fore.RED}Error: Failed to load image at {img_path}{Style.RESET_ALL}")
        return
    else:
        print(f"{Fore.GREEN}Image loaded successfully{Style.RESET_ALL}")
    
    print("\nPREPARING IMAGE")
    print("-"*30)    
    print("Resizing image...")
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    print(f"{Fore.GREEN}Image resized to {IMG_SIZE}x{IMG_SIZE}{Style.RESET_ALL}")
    
    print("Adding batch dimension...")
    img_array = np.array([img_array])
    print(f"{Fore.GREEN}Batch dimension added{Style.RESET_ALL}")
    
    print("Reshaping image...")
    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    print(f"{Fore.GREEN}Image reshaped for model input{Style.RESET_ALL}")
    
    print("Normalizing pixel values...")
    img_array = img_array / 255.0
    print(f"{Fore.GREEN}Image normalized{Style.RESET_ALL}")
    
    print("\nMAKING PREDICTION")
    print("-"*30)
    print("Running model inference...")
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions)
    predicted_label = index_to_label[str(predicted_class)]
    confidence = float(predictions[0][predicted_class]) * 100
    
    print(f"{Fore.GREEN}Prediction complete!{Style.RESET_ALL}")
    print("\n" + "="*50)
    print(f"{Fore.GREEN}RESULT: {Style.BRIGHT}{predicted_label}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Confidence: {Style.BRIGHT}{confidence:.2f}%{Style.RESET_ALL}")
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
    
    print(f"{Fore.CYAN}Creating optimized model architecture...{Style.RESET_ALL}")
    model = Sequential([
        # Simplified architecture with better performance
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(label_to_index), activation='softmax')
    ])

    print(f"{Fore.CYAN}Compiling model with performance optimizations...{Style.RESET_ALL}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"{Fore.CYAN}Setting up training parameters...{Style.RESET_ALL}")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    print(f"{Fore.GREEN}Starting optimized training...{Style.RESET_ALL}")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    
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
    print(f"{Fore.CYAN}Loading model for evaluation...{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Evaluation function called{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}Running evaluation...{Style.RESET_ALL}")
    # ... existing code ...
    
    print(f"{Fore.GREEN}Evaluation Results:{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Accuracy: {Style.BRIGHT}{accuracy:.2f}%{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Loss: {Style.BRIGHT}{loss:.4f}{Style.RESET_ALL}")


def preprocess_image(img_array, IMG_SIZE=100):
    """Standardized image preprocessing"""
    if len(img_array.shape) == 3:  # Color image
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array.astype('float32') / 255.0
    return img_array


if __name__ == "__main__":
    print(f"{Fore.YELLOW}=== Coin Classification System ==={Style.RESET_ALL}")
    print(f"{Fore.CYAN}Select operation mode:{Style.RESET_ALL}")
    
    choice = input("\n1 - Train\n2 - Test\n3 - Evaluate\nEnter: ")
    if choice == "1":
        print(f"{Fore.CYAN}Initializing training, calling train function{Style.RESET_ALL}")
        
        print(f"{Fore.YELLOW}You sure? (y/n): {Style.RESET_ALL}")
        choice = input()
        if choice == "y":
            train()
        else:
            print("Training cancelled")
    elif choice == "2":
        print("\nInitializing testing, calling test function")
        test()
        
    elif choice == "3":
        print("\nCalling evaluation function")
        evaluate()
        
    else:
        print("Invalid choice")