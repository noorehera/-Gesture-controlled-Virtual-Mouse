from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

# Initialize CNN
classifier = Sequential()

# Step 1 - Define input and first layers
classifier.add(Input(shape=(64, 64, 1)))  # Use Input layer
classifier.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))
classifier.add(Dropout(0.5))

# Step 2 - Additional Conv and pooling layers
classifier.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# Step 3 - Flatten and Fully Connected Layers
classifier.add(Flatten())
classifier.add(Dense(units=64, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))
classifier.add(Dense(units=8, activation='softmax'))

# Compile the CNN
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    samplewise_center=True,
    vertical_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create datasets
training_set = train_datagen.flow_from_directory(
    'data/train',
    target_size=(64, 64),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'data/test',
    target_size=(64, 64),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)

# Train the model
classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),  # Dynamically calculate steps
    epochs=10,
    validation_data=test_set,
    validation_steps=len(test_set)  # Dynamically calculate validation steps
)

# Evaluate the model
test_loss, test_accuracy = classifier.evaluate(test_set)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Predicting on a single image
test_img = image.load_img('data/test/fist/1.png', target_size=(64, 64), color_mode='grayscale')
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
result = classifier.predict(test_img)

# Decode prediction
class_indices = training_set.class_indices
class_labels = {v: k for k, v in class_indices.items()}
predicted_label = class_labels[np.argmax(result)]
print(f"Predicted Label: {predicted_label}")

# Save the model
classifier.save('hand_gestures_with_bn.h5')
