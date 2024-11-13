import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Dict, Any
from pipeline import train_and_evaluate_models, load_data


def plot_confusion_matrices(results: Dict[str, Any], dataset_type: str = 'test'):
    """
    Plot confusion matrices for both models side by side.
    """
    # Set up the figure
    plt.figure(figsize=(15, 6))
    
    # Get confusion matrices from results
    xgb_cm = confusion_matrix(
        results['metrics'][dataset_type]['xgboost']['predictions'],
        results['metrics'][dataset_type]['y_true']
    )
    mlp_cm = confusion_matrix(
        results['metrics'][dataset_type]['mlp']['predictions'],
        results['metrics'][dataset_type]['y_true']
    )
    
    # Plot XGBoost confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('XGBoost Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Plot MLP confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(mlp_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('MLP Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.show()

def plot_roc_curves(results: Dict[str, Any], dataset_type: str = 'test'):
    """
    Plot ROC curves for both models.
    """
    plt.figure(figsize=(8, 6))
    
    # Calculate and plot ROC curve for XGBoost
    xgb_fpr, xgb_tpr, _ = roc_curve(
        results['metrics'][dataset_type]['y_true'],
        results['metrics'][dataset_type]['xgboost']['probabilities']
    )
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {results["metrics"][dataset_type]["xgboost"]["roc_auc"]:.3f})')
    
    # Calculate and plot ROC curve for MLP
    mlp_fpr, mlp_tpr, _ = roc_curve(
        results['metrics'][dataset_type]['y_true'],
        results['metrics'][dataset_type]['mlp']['probabilities']
    )
    plt.plot(mlp_fpr, mlp_tpr, label=f'MLP (AUC = {results["metrics"][dataset_type]["mlp"]["roc_auc"]:.3f})')
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pr_curves(results: Dict[str, Any], dataset_type: str = 'test'):
    """
    Plot Precision-Recall curves for both models.
    """
    plt.figure(figsize=(8, 6))
    
    # Calculate and plot PR curve for XGBoost
    xgb_precision, xgb_recall, _ = precision_recall_curve(
        results['metrics'][dataset_type]['y_true'],
        results['metrics'][dataset_type]['xgboost']['probabilities']
    )
    plt.plot(xgb_recall, xgb_precision, 
             label=f'XGBoost (AUC = {results["metrics"][dataset_type]["xgboost"]["pr_auc"]:.3f})')
    
    # Calculate and plot PR curve for MLP
    mlp_precision, mlp_recall, _ = precision_recall_curve(
        results['metrics'][dataset_type]['y_true'],
        results['metrics'][dataset_type]['mlp']['probabilities']
    )
    plt.plot(mlp_recall, mlp_precision, 
             label=f'MLP (AUC = {results["metrics"][dataset_type]["mlp"]["pr_auc"]:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_probability_distributions(results: Dict[str, Any], dataset_type: str = 'test'):
    """
    Plot probability distribution histograms for both models.
    """
    plt.figure(figsize=(15, 5))
    
    # XGBoost probability distribution
    plt.subplot(1, 2, 1)
    for i in range(2):
        mask = results['metrics'][dataset_type]['y_true'] == i
        plt.hist(results['metrics'][dataset_type]['xgboost']['probabilities'][mask], 
                bins=30, alpha=0.5, label=f'Class {i}')
    plt.title('XGBoost Probability Distribution')
    plt.xlabel('Predicted Probability of Class 1')
    plt.ylabel('Count')
    plt.legend()
    
    # MLP probability distribution
    plt.subplot(1, 2, 2)
    for i in range(2):
        mask = results['metrics'][dataset_type]['y_true'] == i
        plt.hist(results['metrics'][dataset_type]['mlp']['probabilities'][mask], 
                bins=30, alpha=0.5, label=f'Class {i}')
    plt.title('MLP Probability Distribution')
    plt.xlabel('Predicted Probability of Class 1')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_model_performance(results: Dict[str, Any], dataset_type: str = 'test'):
    """
    Generate all visualizations for model performance.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results and metrics
    dataset_type : str
        Type of dataset to visualize ('test' or 'validation')
    """
    print(f"\nGenerating visualizations for {dataset_type} set...")
    
    # Plot all visualizations
    plot_confusion_matrices(results, dataset_type)
    plot_roc_curves(results, dataset_type)
    plot_pr_curves(results, dataset_type)
    plot_probability_distributions(results, dataset_type)

# Example usage:
def main():
     # Define paths to your data files
    train_path = '/Users/samuelsetsofia/dev/projects/DTI_Crispy/data/train_data.csv'
    valid_path = '/Users/samuelsetsofia/dev/projects/DTI_Crispy/data/valid_data.csv'
    test_path = '/Users/samuelsetsofia/dev/projects/DTI_Crispy/data/test_data.csv'
    
    # Load data
    print("Loading data...")
    data_dict = load_data(train_path, valid_path, test_path)
    # Your existing code for loading and training models...
    results = train_and_evaluate_models(data_dict)
    
    # Generate visualizations for both validation and test sets
    visualize_model_performance(results, 'validation')
    visualize_model_performance(results, 'test')

if __name__ == "__main__":
    main()