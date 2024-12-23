import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# Dataset directories
MildDemented_dir = r'/kaggle/input/augmented-alzheimer-mri-dataset/AugmentedAlzheimerDataset/MildDemented'
ModerateDemented_dir = r'/kaggle/input/augmented-alzheimer-mri-dataset/AugmentedAlzheimerDataset/ModerateDemented'
NonDemented_dir = r'/kaggle/input/augmented-alzheimer-mri-dataset/AugmentedAlzheimerDataset/NonDemented'
VeryMildDemented_dir = r'/kaggle/input/augmented-alzheimer-mri-dataset/AugmentedAlzheimerDataset/VeryMildDemented'

# Map file paths to class labels
filepaths = []
labels = []
dict_list = [MildDemented_dir, ModerateDemented_dir, NonDemented_dir, VeryMildDemented_dir]
class_labels = ['Mild', 'Moderate', 'Non Demented', 'Very Mild']

for i, j in enumerate(dict_list):
    flist = os.listdir(j)
    for f in flist:
        fpath = os.path.join(j, f)
        filepaths.append(fpath)
        labels.append(class_labels[i])

# Combine file paths and labels into a pandas DataFrame
Fseries = pd.Series(filepaths, name="filepaths")
Lseries = pd.Series(labels, name="labels")
Alzheimer_df = pd.concat([Fseries, Lseries], axis=1)

# Downsample to 7000 samples proportionally
DESIRED_SIZE = 7000
current_size = len(Alzheimer_df)
sampling_fraction = DESIRED_SIZE / current_size

if sampling_fraction < 1.0:
    downsampled_df_list = []
    for label in Alzheimer_df['labels'].unique():
        class_subset = Alzheimer_df[Alzheimer_df['labels'] == label]
        n_to_sample = max(1, int(round(len(class_subset) * sampling_fraction)))
        downsampled_class_subset = class_subset.sample(n=n_to_sample, random_state=42)
        downsampled_df_list.append(downsampled_class_subset)
    Alzheimer_df = pd.concat(downsampled_df_list, ignore_index=True)

Alzheimer_df = Alzheimer_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Downsampled dataset size: {len(Alzheimer_df)}")
print("Class Distribution:\n", Alzheimer_df["labels"].value_counts())

# Split dataset into train, validation, and test sets
train_df, temp_df = train_test_split(
    Alzheimer_df, test_size=0.4, stratify=Alzheimer_df["labels"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["labels"], random_state=42
)

print(f"Training Data: {train_df.shape}")
print(f"Validation Data: {val_df.shape}")
print(f"Testing Data: {test_df.shape}")

# Define the preprocessing function
def preprocess_images(filepaths, target_size=(244, 244)):
    images = []
    for path in filepaths:
        img = load_img(path, target_size=target_size, color_mode='rgb')
        img_array = img_to_array(img)
        images.append(img_array.flatten())
    return np.array(images)

# Define train_and_evaluate_with_validation function
def train_and_evaluate_with_validation(preprocess_func, preprocess_name, train_df, val_df, test_df):
    print(f"\nEvaluating with {preprocess_name}...")
    
    # Preprocess images
    X_train = preprocess_func(train_df["filepaths"].values)
    X_val = preprocess_func(val_df["filepaths"].values)
    X_test = preprocess_func(test_df["filepaths"].values)

    # Encode labels into numerical format
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["labels"])
    y_val = label_encoder.transform(val_df["labels"])
    y_test = label_encoder.transform(test_df["labels"])

    # Standardize the feature vectors
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train k-NN
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    y_val_pred = knn_model.predict(X_val)
    y_test_pred = knn_model.predict(X_test)

    # Evaluate on validation set
    print(f"\nValidation Report ({preprocess_name}):")
    print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

    # Evaluate on test set
    print(f"\nTest Report ({preprocess_name}):")
    print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

    # Plot confusion matrix for validation set
    cm_val = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Validation Confusion Matrix ({preprocess_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Plot confusion matrix for test set
    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Test Confusion Matrix ({preprocess_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Final accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy with {preprocess_name}: {test_accuracy * 100:.2f}%")

# Evaluate using resized images
train_and_evaluate_with_validation(
    lambda x: preprocess_images(x, target_size=(180, 180)),
    "180x180 original images",
    train_df,
    val_df,
    test_df
)

# Evaluate using resized images
train_and_evaluate_with_validation(
    lambda x: preprocess_images(x, target_size=(64, 64)),
    "64x64 original images",
    train_df,
    val_df,
    test_df
)



def preprocess_images_with_optional_pca(
    filepaths, labels, target_size=(180, 180), use_pca=False, n_components=None, pca=None
):
    images_list = []
    for fp in filepaths:
        img = load_img(fp, color_mode='rgb')
        if target_size is not None:
            if isinstance(target_size, tuple):
                img = img.resize(target_size)
            else:
                img = img.resize((target_size, target_size))
        img_array = img_to_array(img)
        images_list.append(img_array)
    images_array = np.array(images_list, dtype=np.float32)
    images_flat = images_array.reshape(len(images_array), -1)
    if use_pca:
        scaler = StandardScaler()
        images_flat_scaled = scaler.fit_transform(images_flat)
        if pca is None:
            pca_temp = PCA().fit(images_flat_scaled)
            cum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            optimal_n = np.argmax(cum_var >= 0.97) + 1
            print(f"Number of components for 95% variance: {optimal_n}")
            pca = PCA(n_components=optimal_n, random_state=42)
            images_features = pca.fit_transform(images_flat_scaled)
        else:
            images_features = pca.transform(images_flat_scaled)
    else:
        images_features = images_flat
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    return images_features, labels_encoded, label_encoder.classes_

def train_and_evaluate(preprocess_func, preprocess_name, train_df, test_df):
    print(f"\nEvaluating with {preprocess_name}...")
    X_train, y_train, classes = preprocess_func(train_df['filepaths'].values, train_df['labels'].values)
    X_test, y_test, _ = preprocess_func(test_df['filepaths'].values, test_df['labels'].values)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    print(f"\nClassification Report ({preprocess_name}):")
    print(classification_report(y_test, y_pred, target_names=classes))
    print("\nConfusion Matrix ({preprocess_name}):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix ({preprocess_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy with {preprocess_name}: {accuracy * 100:.2f}%")

# Train and evaluate
train_and_evaluate(preprocess_images_with_optional_pca, "PCA applied images", train_df, test_df)