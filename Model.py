# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:45:57 2025

@author: CMP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

dataset_path = 'metadata.csv'
df = pd.read_csv(dataset_path)

# Display basic dataset details
df.head()
df.tail()
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
print("Duplicated Rows:", df.duplicated().sum())
print("Missing Values:")
print(df.isnull().sum())
df.info()
print("Unique Values Per Column:")
print(df.nunique())

# Classifying Features
def classify_features(df):
    categorical_features = []
    non_categorical_features = []
    discrete_features = []
    continuous_features = []
    
    for column in df.columns:
        if df[column].dtype == 'object':
            if df[column].nunique() < 10:
                categorical_features.append(column)
            else:
                non_categorical_features.append(column)
        elif df[column].dtype in ['int64', 'float64']:
            if df[column].nunique() < 10:
                discrete_features.append(column)
            else:
                continuous_features.append(column)
    
    return categorical_features, non_categorical_features, discrete_features, continuous_features

categorical, non_categorical, discrete, continuous = classify_features(df)
print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)

df.fillna("Not Available", inplace=True)

for col in categorical:
    print(f"{col}: {df[col].unique()}\n")
    
    
for col in categorical:
    print(df[col].value_counts())
    print()    


# Countplots for categorical features
for col in categorical:
    plt.figure(figsize=(15,6))
    sns.countplot(x=df[col], palette='hls')
    plt.title(f"Distribution of {col}")
    plt.show()
    
    
# Pie charts for categorical features
for col in categorical:
    plt.figure(figsize=(10,7))
    plt.pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct='%1.1f%%', textprops={'fontsize': 12})
    plt.title(f"Proportion of {col}")
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

for col in discrete + continuous:
    plt.figure(figsize=(12,5))
    sns.distplot(df[col], hist=True, kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=90)
    plt.show()
    
# Boxplots for numerical features
for col in discrete + continuous:
    plt.figure(figsize=(12,5))
    sns.boxplot(x=df[col], palette='hls')
    plt.title(f"Boxplot of {col}")
    plt.xticks(rotation=90)
    plt.show()

# Violin plots for numerical features
for col in discrete + continuous:
    plt.figure(figsize=(12,5))
    sns.violinplot(x=df[col], palette='hls')
    plt.title(f"Violin Plot of {col}")
    plt.xticks(rotation=90)
    plt.show()
from sklearn.model_selection import train_test_split

real_df = df[df["label"] == "REAL"]
fake_df = df[df["label"] == "FAKE"]

sample_size = min(len(real_df), len(fake_df))
real_df = real_df.sample(sample_size, random_state=42)
fake_df = fake_df.sample(sample_size, random_state=42)

sample_meta = pd.concat([real_df, fake_df])

Train_set, Test_set = train_test_split(sample_meta, test_size=0.2, random_state=42, stratify=sample_meta['label'])
Train_set, Val_set  = train_test_split(Train_set, test_size=0.3, random_state=42, stratify=Train_set['label'])
print("Train, Validation, and Test Set Sizes:", Train_set.shape, Val_set.shape, Test_set.shape)


import cv2
import os
image_path = 'faces_224/'
image_files = sorted(os.listdir(image_path))
selected_images = image_files[:9]

plt.figure(figsize=(10, 10))
for index, image_file in enumerate(selected_images):
    image = cv2.imread(os.path.join(image_path, image_file))
    plt.subplot(3, 3, index + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Image {index + 1}')
    plt.axis('off')
plt.show()

# Display image resolutions
for i, image_file in enumerate(image_files[:10]):
    image = cv2.imread(os.path.join(image_path, image_file))
    if image is not None:
        height, width, _ = image.shape
        print(f"Resolution of image {i+1}: {width} x {height}")
    else:
        print(f"Error reading image {i+1}")
# Visualizing a batch of real vs fake images
plt.figure(figsize=(15,15))
# for cur, i in enumerate(Train_set.index[25:50]):
#     plt.subplot(5, 5, cur+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(cv2.imread(image_path + Train_set.loc[i,'videoname'][:-4] + '.jpg'))
#     plt.xlabel('FAKE Image' if Train_set.loc[i,'label']=='FAKE' else 'REAL Image')
# plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Paths
dataset_path = 'metadata.csv'
image_path = 'faces_224/'
IMG_SIZE = 224

# Load metadata
df = pd.read_csv(dataset_path)

# Basic cleaning
df.drop_duplicates(inplace=True)
df.fillna("Not Available", inplace=True)

# Balance the dataset
real_df = df[df["label"] == "REAL"]
fake_df = df[df["label"] == "FAKE"]
sample_size = min(len(real_df), len(fake_df))
real_df = real_df.sample(sample_size, random_state=42)
fake_df = fake_df.sample(sample_size, random_state=42)
sample_meta = pd.concat([real_df, fake_df])

# Train/Val/Test split
Train_set, Test_set = train_test_split(sample_meta, test_size=0.2, random_state=42, stratify=sample_meta['label'])
Train_set, Val_set  = train_test_split(Train_set, test_size=0.3, random_state=42, stratify=Train_set['label'])

print("Train:", Train_set.shape, "Val:", Val_set.shape, "Test:", Test_set.shape)

# Load and preprocess images
def load_image_from_row(row):
    image_name = row['videoname'].replace('.mp4', '.jpg')
    image_file = os.path.join(image_path, image_name)
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image not found: {image_file}")
    img = cv2.imread(image_file)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    label = 1 if row['label'] == 'FAKE' else 0
    return img, label

def prepare_dataset(df):
    images = []
    labels = []
    for _, row in df.iterrows():
        try:
            img, label = load_image_from_row(row)
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Skipping row due to error: {e}")
    return np.array(images), np.array(labels)

# Load all datasets
X_train, y_train = prepare_dataset(Train_set)
X_val, y_val = prepare_dataset(Val_set)
X_test, y_test = prepare_dataset(Test_set)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_val, y_val))

# Save the model
model.save("face_classifier_model.h5")
print("✅ Model saved as face_classifier_model.h5")

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

