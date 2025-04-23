
from google.colab import drive
import imgaug.augmenters as iaa
import cv2
import os
import numpy as np

# Mount Google Drive
drive.mount('/content/drive')

import os

# Path to the root directory where your fruit folders are located
input_directory = '/content/drive/MyDrive/Original Image/Fruits Original'
output_directory = '/content/drive/MyDrive/FruitDataset_Augmented_New'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Augmentation techniques
augmenters = [
    iaa.Affine(rotate=(45)),               # 45-degree Rotation
    iaa.Affine(rotate=(60)),               # 60-degree Rotation
    iaa.Affine(rotate=(90)),               # 90-degree Rotation
    iaa.Fliplr(0.5),                       # Horizontal Flip
    iaa.Affine(scale=(0.8, 1.2)),          # Zooming
    iaa.Multiply((0.8, 1.2)),              # Brightness Adjustment
    iaa.Affine(shear=(-16, 16)),           # Shearing
    iaa.AdditiveGaussianNoise(scale=0.05*255)  # Adding Gaussian Noise
]

# Function to apply augmentations and save images
def augment_and_save(image_path, save_path):
    image = cv2.imread(image_path)
    for i, aug in enumerate(augmenters):
        augmented_image = aug(image=image)
        file_name = f"{os.path.basename(image_path).split('.')[0]}_aug_{i+1}.jpg"
        cv2.imwrite(os.path.join(save_path, file_name), augmented_image)

# Process each folder and subfolder
for fruit_folder in os.listdir(input_directory):
    fruit_path = os.path.join(input_directory, fruit_folder)
    output_fruit_path = os.path.join(output_directory, fruit_folder)

    if not os.path.exists(output_fruit_path):
        os.makedirs(output_fruit_path)

    for fruit_type_folder in os.listdir(fruit_path):
        fruit_type_path = os.path.join(fruit_path, fruit_type_folder)
        output_type_path = os.path.join(output_fruit_path, fruit_type_folder)

        if not os.path.exists(output_type_path):
            os.makedirs(output_type_path)

        for image_file in os.listdir(fruit_type_path):
            image_file_path = os.path.join(fruit_type_path, image_file)
            augment_and_save(image_file_path, output_type_path)

import cv2
import os

# Paths to the augmented images directory and the output directory for resized images
augmented_directory = '/content/drive/MyDrive/FruitDataset_Augmented_New'
resized_directory = '/content/drive/MyDrive/FruitDataset_Resized'

# Desired dimensions for resizing (e.g., 256x256 pixels)
resize_width = 512
resize_height = 512

# Create output directory if it doesn't exist
if not os.path.exists(resized_directory):
    os.makedirs(resized_directory)

# Function to resize images and save them
def resize_and_save(image_path, save_path, width, height):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite(save_path, resized_image)

# Process each folder and subfolder
for fruit_folder in os.listdir(augmented_directory):
    fruit_path = os.path.join(augmented_directory, fruit_folder)
    output_fruit_path = os.path.join(resized_directory, fruit_folder)

    if not os.path.exists(output_fruit_path):
        os.makedirs(output_fruit_path)

    for fruit_type_folder in os.listdir(fruit_path):
        fruit_type_path = os.path.join(fruit_path, fruit_type_folder)
        output_type_path = os.path.join(output_fruit_path, fruit_type_folder)

        if not os.path.exists(output_type_path):
            os.makedirs(output_type_path)

        for image_file in os.listdir(fruit_type_path):
            image_file_path = os.path.join(fruit_type_path, image_file)
            save_path = os.path.join(output_type_path, image_file)
            resize_and_save(image_file_path, save_path, resize_width, resize_height)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import ultralytics
from ultralytics import YOLO
from PIL import Image
import glob

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Paths configuration
resized_directory = '/content/drive/MyDrive/FruitDataset_Resized'
output_directory = '/content/drive/MyDrive/FruitModelOutputs'
# Create output directories if they don't exist
os.makedirs(output_directory, exist_ok=True)

# Parameters
IMG_SIZE = (512, 512)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = len(os.listdir(resized_directory))  # Automatically detect number of classes

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    resized_directory,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    resized_directory,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Get class names
class_names = list(train_generator.class_indices.keys())

## 1. CNN Model
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Build and train CNN
cnn_model = build_cnn_model((IMG_SIZE[0], IMG_SIZE[1], 3), NUM_CLASSES)
cnn_history = cnn_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(os.path.join(output_directory, 'best_cnn_model.h5'), save_best_only=True)
    ]
)

# Save CNN model
cnn_model.save(os.path.join(output_directory, 'fruit_cnn_model.h5'))

## 2. MobileNetV2 Model
def build_mobilenet_model(input_shape, num_classes):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    # Freeze the base model
    base_model.trainable = False

    model = Sequential([
        base_model,
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Build and train MobileNetV2
mobilenet_model = build_mobilenet_model((IMG_SIZE[0], IMG_SIZE[1], 3), NUM_CLASSES)
mobilenet_history = mobilenet_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(os.path.join(output_directory, 'best_mobilenet_model.h5'), save_best_only=True)
    ]
)

# Save MobileNetV2 model
mobilenet_model.save(os.path.join(output_directory, 'fruit_mobilenet_model.h5'))

## 3. YOLOv8 Model (Object Detection)
# Note: YOLOv8 requires annotation files in YOLO format
# This is a simplified version assuming you have YOLO format annotations

def train_yolov8_model(data_yaml_path, epochs=20):
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=IMG_SIZE[0],
        batch=BATCH_SIZE,
        project=os.path.join(output_directory, 'yolov8'),
        name='fruit_detection'
    )

    return model

## Evaluation Functions
def plot_confusion_matrix(model, generator, class_names, model_name):
    # Get predictions
    predictions = model.predict(generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = generator.classes

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(output_directory, f'confusion_matrix_{model_name}.png'))
    plt.show()

    # Classification report
    print(f"Classification Report - {model_name}:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def plot_auc_roc(model, generator, class_names, model_name):
    # Get predictions
    predictions = model.predict(generator)
    y_true = generator.classes

    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
                   'purple', 'pink', 'brown', 'gray', 'olive'])

    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-class ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_directory, f'roc_auc_{model_name}.png'))
    plt.show()

    return roc_auc

def plot_training_history(history, model_name):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(os.path.join(output_directory, f'training_history_{model_name}.png'))
    plt.show()

## Evaluate Models
# CNN Evaluation
plot_training_history(cnn_history, 'CNN')
cnn_roc_auc = plot_auc_roc(cnn_model, val_generator, class_names, 'CNN')
plot_confusion_matrix(cnn_model, val_generator, class_names, 'CNN')

# MobileNetV2 Evaluation
plot_training_history(mobilenet_history, 'MobileNetV2')
mobilenet_roc_auc = plot_auc_roc(mobilenet_model, val_generator, class_names, 'MobileNetV2')
plot_confusion_matrix(mobilenet_model, val_generator, class_names, 'MobileNetV2')

# Print AUC values
print("\nAUC Values:")
print("CNN Model:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {cnn_roc_auc[i]:.4f}")

print("\nMobileNetV2 Model:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {mobilenet_roc_auc[i]:.4f}")

yolo_model = train_yolov8_model('/content/drive/MyDrive/FruitDataset_Resized/fruit_label.yaml')

results_path = '/content/drive/MyDrive/FruitDataset_Resized/yolov8_output.csv'
df = pd.read_csv(results_path)

plt.figure(figsize=(12, 5))
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training Curves')
plt.legend()
plt.grid()
plt.show()

# Run prediction on test set
metrics = model.val(split='test', save_json=True, conf=0.25)

# Get predicted and true labels
pred_labels = metrics.results_dict['pred']  # predicted labels
true_labels = metrics.results_dict['gt']    # ground truth labels

# Generate confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.names.values(), yticklabels=model.names.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print classification report
print(classification_report(true_labels, pred_labels, target_names=model.names.values()))

# AUC (one-vs-rest for multiclass)
true_onehot = np.eye(len(model.names))[true_labels]
pred_onehot = np.eye(len(model.names))[pred_labels]
auc = roc_auc_score(true_onehot, pred_onehot, average="macro", multi_class="ovr")
print(f"AUC (macro): {auc:.4f}")
