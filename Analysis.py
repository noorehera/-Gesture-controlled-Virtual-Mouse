import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import numpy as np

# Define paths
train_dir = "/Users/swetha/Desktop/sem6/ml project/gesture/data/train"
val_dir = "/Users/swetha/Desktop/sem6/ml project/gesture/data/validation"

# Image data generators
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load datasets
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

val_dataset = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for validation dataset
)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_dataset.class_indices), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Evaluate the model
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(val_dataset)

# Predictions
predictions = model.predict(val_dataset, steps=len(val_dataset), verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# True labels
true_classes = val_dataset.classes

# F1-score calculation
f1 = f1_score(true_classes, predicted_classes, average='weighted')

# Classification report
target_names = list(val_dataset.class_indices.keys())
classification_report_str = classification_report(true_classes, predicted_classes, target_names=target_names)

# Print metrics
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1-Score: {f1}")

print("\nClassification Report:")
print(classification_report_str)
