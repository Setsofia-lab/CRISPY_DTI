import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_data(train_path, valid_path, test_path):
    """
    Load and prepare train, validation and test datasets.
    
    Parameters:
    -----------
    train_path : str
        Path to training data CSV file
    valid_path : str
        Path to validation data CSV file
    test_path : str
        Path to test data CSV file
        
    Returns:
    --------
    dict : Dictionary containing features and labels for each dataset split
    """
    # Load datasets
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features and target
    X_train = train_df.drop('Label', axis=1)
    y_train = train_df['Label']
    
    X_valid = valid_df.drop('Label', axis=1)
    y_valid = valid_df['Label']
    
    X_test = test_df.drop('Label', axis=1)
    y_test = test_df['Label']
    
    # Return dictionary containing all splits
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_valid': X_valid,
        'y_valid': y_valid,
        'X_test': X_test,
        'y_test': y_test
    }

def evaluate_model(model, X, y, model_name, scaler=None):
    """
    Evaluate model performance and print metrics.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to evaluate
    X : array-like
        Features
    y : array-like
        True labels
    model_name : str
        Name of the model for printing
    scaler : StandardScaler, optional
        Scaler for normalizing features if needed
    """
    # Scale features if scaler is provided
    if scaler is not None:
        X = scaler.transform(X)
    
    # Get predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Print evaluation metrics
    print("\n" + "="*50)
    print(f"{model_name} Evaluation")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    # Calculate ROC AUC and PR AUC
    roc_auc = roc_auc_score(y, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\nROC AUC Score: {roc_auc:.3f}")
    print(f"PR AUC Score: {pr_auc:.3f}")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def train_and_evaluate_models(data_dict, random_state=42):
    """
    Train and evaluate XGBoost and MLP models.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing train, validation and test datasets
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : Trained XGBoost and MLP models
    """
    # Extract data
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_valid = data_dict['X_valid']
    y_valid = data_dict['y_valid']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    # Scale features for MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Initialize models
    xgb_model = XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=500,
        random_state=random_state
    )
    
    # Train models
    print("Training XGBoost model...")
    xgb_model.fit(X_train, y_train)
   
    print("\nTraining Set Evaluation XGBoost:")
    xgb_train_metrics = evaluate_model(xgb_model, X_train, y_train, "XGBoost")
    
    
    print("\nTraining MLP model...")
    mlp_model.fit(X_train_scaled, y_train)


    print("\nTraining Set Evaluation MLP:")
    mlp_train_metrics = evaluate_model(mlp_model, X_valid, y_valid, "MLP", scaler)

    # Evaluate on validation set
    print("\nValidation Set Evaluation:")
    xgb_valid_metrics = evaluate_model(xgb_model, X_valid, y_valid, "XGBoost")
    mlp_valid_metrics = evaluate_model(mlp_model, X_valid, y_valid, "MLP", scaler)
    
    # Evaluate on test set
    print("\nTest Set Evaluation:")
    xgb_test_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    mlp_test_metrics = evaluate_model(mlp_model, X_test, y_test, "MLP", scaler)
    
    return {
        'models': {
            'xgboost': xgb_model,
            'mlp': mlp_model
        },
        'scaler': scaler,
        'metrics': {
            'training': {
                'xgboost': xgb_train_metrics,
                'mlp': mlp_train_metrics
            },
            'validation': {
                'xgboost': xgb_valid_metrics,
                'mlp': mlp_valid_metrics
            },
            'test': {
                'xgboost': xgb_test_metrics,
                'mlp': mlp_test_metrics
            }
        }
    }

def main():
    # Define paths to your data files
    train_path = '/Users/samuelsetsofia/dev/projects/DTI_Crispy/data/train_data.csv'
    valid_path = '/Users/samuelsetsofia/dev/projects/DTI_Crispy/data/valid_data.csv'
    test_path = '/Users/samuelsetsofia/dev/projects/DTI_Crispy/data/test_data.csv'
    
    # Load data
    print("Loading data...")
    data_dict = load_data(train_path, valid_path, test_path)
    
    # Train and evaluate models
    print("Training and evaluating models...")
    results = train_and_evaluate_models(data_dict)
    
    # Access trained models and results
    xgb_model = results['models']['xgboost']
    mlp_model = results['models']['mlp']
    scaler = results['scaler']
    
    # Print final validation and test metrics
    print("\nTraining Metrics:")
    print(f"XGBoost ROC AUC: {results['metrics']['training']['xgboost']['roc_auc']:.3f}")
    print(f"MLP ROC AUC: {results['metrics']['training']['mlp']['roc_auc']:.3f}")

    print("\nFinal Validation Metrics:")
    print(f"XGBoost ROC AUC: {results['metrics']['validation']['xgboost']['roc_auc']:.3f}")
    print(f"MLP ROC AUC: {results['metrics']['validation']['mlp']['roc_auc']:.3f}")
    
    print("\nFinal Test Metrics:")
    print(f"XGBoost ROC AUC: {results['metrics']['test']['xgboost']['roc_auc']:.3f}")
    print(f"MLP ROC AUC: {results['metrics']['test']['mlp']['roc_auc']:.3f}")

if __name__ == "__main__":
    main()