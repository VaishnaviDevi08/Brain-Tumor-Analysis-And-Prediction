# -- coding: utf-8 --
"""
@author: Serena Grazia De Benedictis, Grazia Gargano, Gaetano Settembre
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import time
import tensorly as tl
from utils import compute_tucker_decomposition, import_data, data_to_negative, display_image_grid

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Set Training and Test set path
path_train = 'Dataset/Training'
path_test = 'Dataset/Testing'

# Import Training and Test Set
x_train, y_train = import_data(path_train, labels, 250)
x_test, y_test = import_data(path_test, labels, 250)

# Make images negative
x_train = data_to_negative(x_train)
x_test = data_to_negative(x_test)

# Normalize images
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0

# Display some training images
images_to_display = [x_train[i] for i in range(36)]
display_image_grid(images_to_display, grid_size=(6, 6), figsize=(10, 10), cmap="gray", title="Image Grid of data")

# Flatten images and prepare dataset matrices
X_train = np.vstack([image.flatten().astype(np.float32) for image in x_train])
X_test = np.vstack([image.flatten().astype(np.float32) for image in x_test])

# Apply PCA for dimensionality reduction
n_components = 300
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Scale PCA features
scaler = StandardScaler()
X_train_pca = scaler.fit_transform(X_train_pca)
X_test_pca = scaler.transform(X_test_pca)

# Define model constructors with parameter tuning
model_constructors = {
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "SVM (Linear)": lambda: LinearSVC(max_iter=5000, class_weight='balanced'),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=100, random_state=42),
    "XGBoost": lambda: XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, random_state=42)
}

# Metrics for evaluation
metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F1 Score": f1_score
}

# Function to train and evaluate models
def evaluate_models(X_train, X_test, y_train, y_test, feature_type="PCA"):
    for name, model_constructor in model_constructors.items():
        print(f"Training {name} with {feature_type} features...")
        
        scores = {metric_name: [] for metric_name in metrics}
        train_times = []

        for _ in range(5):  # 5 runs to average performance
            model = model_constructor()
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            y_pred = model.predict(X_test)

            train_times.append(training_time)
            for metric_name, metric_func in metrics.items():
                if metric_name == "Accuracy":
                    score = metric_func(y_test, y_pred)
                else:
                    score = metric_func(y_test, y_pred, average='weighted')
                scores[metric_name].append(score)

        # Display results
        avg_scores = {metric_name: np.mean(score_list) for metric_name, score_list in scores.items()}
        std_scores = {metric_name: np.std(score_list) for metric_name, score_list in scores.items()}
        avg_times = np.mean(train_times)
        std_times = np.std(train_times)

        print(f"Average Performance of {name}:")
        for metric_name, avg_score in avg_scores.items():
            print(f"{metric_name}: {avg_score:.4f} ± {std_scores[metric_name]:.4f}")
        print(f"Training time stats: {avg_times:.4f} ± {std_times:.4f}")
        print(f"{name} finished training.\n")

# Evaluate on PCA features
evaluate_models(X_train_pca, X_test_pca, y_train, y_test)

# Tensor decomposition
x_train_fold = tl.fold(X_train, mode=2, shape=(250, 250, x_train.shape[0]))
x_test_fold = tl.fold(X_test, mode=2, shape=(250, 250, x_test.shape[0]))
rank_decomposition = (30, 30, 300)
core, U0, U1, U2 = compute_tucker_decomposition(rank_decomposition, x_train_fold)

# Project using Tucker
pU2 = core @ U2.T
core_p = tl.unfold(pU2, mode=2)

# Build test features using decomposition
x_test_reprojected = tl.tenalg.mode_dot(tl.tenalg.mode_dot(x_test_fold, np.linalg.pinv(U0), mode=0), np.linalg.pinv(U1), mode=1)
x_test_unfolded = tl.unfold(x_test_reprojected, mode=2)

# Scale Tucker features
core_p_scaled = scaler.fit_transform(core_p)
x_test_unfolded_scaled = scaler.transform(x_test_unfolded)

# Evaluate on Tucker features
evaluate_models(core_p_scaled, x_test_unfolded_scaled, y_train, y_test, feature_type="Tucker Decomposition")
