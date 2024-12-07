from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

# Initializing CNN
classifier = Sequential()

# Step 1 - Convolution and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu', padding='same'))
classifier.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))
classifier.add(Dropout(0.5))

# Adding 2nd Convolution and pooling
classifier.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# Step 3 - Flatten
classifier.add(Flatten())

# Creating ANN
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=8, activation='softmax'))

# Compile the CNN
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    samplewise_center=True,
    vertical_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

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

# Fitting the model
classifier.fit(
    training_set,
    steps_per_epoch=7999,
    epochs=1,
    validation_data=test_set,
    validation_steps=4000
)

# Evaluating the model on the test set
classifier.evaluate(test_set)

# Load a specific test image for prediction
test_img = image.load_img('data/test/fist/1.png', target_size=(64, 64), color_mode='grayscale')
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
result = classifier.predict(test_img)
print(result)

# Saving the model
classifier.save('hand_gestures_1.h5')
