import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

# Helper function for plotting confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix with seaborn heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Helper function for printing evaluation metrics
def print_metrics(y_true, y_pred, title):
    """Print evaluation metrics."""
    print(f"\n{'-'*50}")
    print(f"{title} Results:")
    print(f"{'-'*50}")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define and train XGBoost model
xgb_params = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'reg_lambda': 1.5,
    'gamma': 0.1,
    'min_child_weight': 25,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

xgb_model = XGBClassifier(**xgb_params, random_state=42)
print("\nTraining XGBoost model...")
xgb_model.fit(X_train, y_train)

# Make predictions with XGBoost
xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

# Evaluate XGBoost results
print_metrics(y_train, xgb_train_pred, 'XGBoost - Training Set')
plot_confusion_matrix(y_train, xgb_train_pred, 'XGBoost - Training Set')
print_metrics(y_test, xgb_test_pred, 'XGBoost - Test Set')
plot_confusion_matrix(y_test, xgb_test_pred, 'XGBoost - Test Set')

# Define and train MLP model
mlp_params = {
    'hidden_layer_sizes': (100,),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 200,
    'random_state': 42
}

mlp_model = MLPClassifier(**mlp_params)
print("\nTraining MLP model...")
mlp_model.fit(X_train, y_train)

# Make predictions with MLP
mlp_train_pred = mlp_model.predict(X_train)
mlp_test_pred = mlp_model.predict(X_test)

# Evaluate MLP results
print_metrics(y_train, mlp_train_pred, 'MLP - Training Set')
plot_confusion_matrix(y_train, mlp_train_pred, 'MLP - Training Set')
print_metrics(y_test, mlp_test_pred, 'MLP - Test Set')
plot_confusion_matrix(y_test, mlp_test_pred, 'MLP - Test Set')