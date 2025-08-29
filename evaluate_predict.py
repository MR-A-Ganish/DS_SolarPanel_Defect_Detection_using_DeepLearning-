import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load your trained model
model_path = "solar_panel_defect_model.h5"
model = load_model(model_path)
print(f"✅ Model loaded from: {model_path}")
print(f"📐 Expected input shape: {model.input_shape}")

# Define class labels (same order as training)
class_labels = ["bird-drop", "clean", "dusty", "electrical-damage", "physical-damage"]

# Folder containing test images
test_folder = r"C:\Users\asus\Desktop\guvi\PROJECT-5\test_images"

# Walk through all subfolders and files
for root, dirs, files in os.walk(test_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            try:
                # Load and preprocess image to match model input
                img = image.load_img(img_path, target_size=(128, 128))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Predict
                prediction = model.predict(img_array)
                predicted_class = class_labels[np.argmax(prediction)]

                print(f"🖼️ Image: {file} → Predicted: {predicted_class}")

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
        else:
            print(f"⚠️ Skipped non-image file: {file}")