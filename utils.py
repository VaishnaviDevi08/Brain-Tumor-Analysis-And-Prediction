"""
@author: Serena Grazia De Benedictis, Grazia Gargano, Gaetano Settembre
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from tensorly.decomposition import tucker

def import_data(dataset_path, labels, image_size):
    x_data = []  # List to hold the images
    y_data = []  # List to hold the corresponding labels

    for label in labels:
        label_path = os.path.join(dataset_path, label)

        print(f"Checking path: {label_path}")
        if not os.path.exists(label_path):
            print(f"Error: The path '{label_path}' does not exist.")
            continue
        if not os.path.isdir(label_path):
            print(f"Error: The path '{label_path}' is not a directory.")
            continue

        label_files = os.listdir(label_path)
        if len(label_files) == 0:
            print(f"Error: The directory '{label_path}' is empty.")
            continue

        print(f"Processing label: {label}, Path: {label_path}")

        for file in tqdm(label_files, desc=f"Processing {label} files"):
            file_path = os.path.join(label_path, file)

            if os.path.isdir(file_path) or not file.lower().endswith('.jpg'):
                continue

            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Unable to read image {file} in {label_path}")
                continue

            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=80)
            image = cv2.resize(image, (image_size, image_size))

            x_data.append(image)
            y_data.append(labels.index(label))

    if not x_data or not y_data:
        print("Error: No images were loaded. Check dataset paths and files.")
        return [], []

    return x_data, y_data


def import_data_without_preprocessing(dataset_path, labels, image_size):
    x_data, y_data = [], []

    for label in labels:
        label_path = os.path.join(dataset_path, label)

        if not os.path.exists(label_path):
            print(f"Error: The path '{label_path}' does not exist.")
            continue

        if not os.path.isdir(label_path):
            print(f"Error: The path '{label_path}' is not a directory.")
            continue

        for file in tqdm(os.listdir(label_path), desc=f"Processing {label} files"):
            file_path = os.path.join(label_path, file)

            if os.path.isdir(file_path) or not file.lower().endswith(('jpg', 'jpeg', 'png')):
                continue

            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Unable to read image {file} in {label_path}")
                continue

            image = cv2.resize(image, (image_size, image_size))
            x_data.append(image)
            y_data.append(labels.index(label))

    return x_data, y_data


def data_to_negative(data):
    return 255 - np.array(data)


def display_image_grid(images, grid_size=(6, 6), figsize=(10, 10), cmap="gray", title=None):
    total_images = grid_size[0] * grid_size[1]

    if len(images) != total_images:
        raise ValueError(f"Number of images ({len(images)}) does not match the grid size {grid_size}.")

    fig, axes = plt.subplots(*grid_size, figsize=figsize)
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def fit_predict_classifier(model, x_train, y_train, x_test):
    start_time = time.time()
    model.fit(x_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"Model trained in {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
    y_pred = model.predict(x_test)
    return y_pred


def evaluate_classifier(model, y_test, y_pred):
    classifier_name = type(model).__name__
    print(f"Evaluating classifier {classifier_name}")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'F1-score: {f1}')
    print(classification_report(y_test, y_pred, digits=4))


def plot_confusion_matrix(conf_matrix, classes):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    disp.plot(colorbar=False, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()


def compute_tucker_decomposition(rank, x_data):
    start_time = time.time()
    core, factors = tucker(x_data, rank)
    elapsed_time = time.time() - start_time
    print(f"Time Tucker decomp.: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
    return core, factors[0], factors[1], factors[2]


def plot_rank_size(model, f1_list, accuracy_list):
    plt.plot([10, 20, 30, 40, 50], f1_list, 'o-', label='F1-score')
    plt.plot([10, 20, 30, 40, 50], accuracy_list, 'o-', label='Accuracy')
    plt.xlabel('Dim. mode1, mode2 core')
    plt.ylabel('Metrics values')
    plt.legend()
    plt.title(f"{type(model).__name__} performances after Tucker decomp.")
    plt.show()
