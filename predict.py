import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model_path = 'skin_disease_model.keras'
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully")

# Class labels (must match the folder names used during training)
class_labels = ['Chronic Dermatitis', 'Lichen Planus', 'Pityriasis Rosea', 'Psoriasis', 'Seborrheic Dermatitis']

# Function to preprocess the image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the input shape of the model
        img_array /= 255.0  # Normalize to [0, 1]
        print("Image successfully preprocessed.")
        return img_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# Function to predict the class of the image
def predict_skin_disease(img_path):
    img_array = preprocess_image(img_path)
    if img_array is None:
        return "Error during preprocessing"
    
    try:
        prediction = model.predict(img_array)
        print(f"Model prediction: {prediction}")
        predicted_class = np.argmax(prediction, axis=1)
        disease_name = class_labels[predicted_class[0]]
        print(f"Predicted class: {disease_name}")
        return disease_name
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction"

# Main execution block
if __name__ == "__main__":
    # You can either hardcode the image path
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    
    # Or take it as a command line argument
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist")
        sys.exit(1)
        
    # Make prediction
    result = predict_skin_disease(image_path)
    print("\nFinal Prediction:", result)






    