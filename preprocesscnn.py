import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -------- Paths --------
BASE_DIR = r"C:\Users\asus\Desktop\guvi\PROJECT-5"   # Change if dataset is in another path

if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(f"Dataset path not found: {BASE_DIR}")

# -------- Data Preprocessing --------
img_size = (128, 128)   # Resize all images
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# -------- CNN Model --------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')  # Output layer matches no. of classes
])

# -------- Compile --------
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------- Train --------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# -------- Save Model --------
model.save("solar_panel_defect_model.h5")

print("✅ Training completed. Model saved as solar_panel_defect_model.h5")
