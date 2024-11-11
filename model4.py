import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

class XGBoostClassifier:
    def __init__(self, params=None):
        # Default parameters with balanced class weights and stronger regularization
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'error'],
            'learning_rate': 0.01,
            'max_depth': 4,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 2,
            'reg_alpha': 1,
            'gamma': 0.2,
            'scale_pos_weight': 1,  # Will be calculated based on data
            'tree_method': 'hist',  # Faster tree construction
            'random_state': 42
        }
        self.params = params if params is not None else self.default_params
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self, X, y=None, is_training=True):
        """Preprocess features with standardization"""
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        if y is not None:
            return X_scaled, y
        return X_scaled
    
    def train(self, X_train, y_train, X_val=None, y_val=None, num_rounds=1000):
        """Train the XGBoost model with early stopping"""
        # Calculate scale_pos_weight
        self.params['scale_pos_weight'] = np.sum(y_train == 0) / np.sum(y_train == 1)
        
        # Preprocess data
        X_train_scaled, y_train = self.preprocess_data(X_train, y_train)
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val = self.preprocess_data(X_val, y_val, is_training=False)
            eval_set = [(X_val_scaled, y_val)]
        else:
            eval_set = None
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        if eval_set:
            dval = xgb.DMatrix(X_val_scaled, label=y_val)
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
        else:
            watchlist = [(dtrain, 'train')]
        
        # Train model with early stopping
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_rounds,
            evals=watchlist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return self
    
    def predict(self, X, threshold=0.5):
        """Make predictions with custom threshold"""
        X_scaled = self.preprocess_data(X, is_training=False)
        dtest = xgb.DMatrix(X_scaled)
        probas = self.model.predict(dtest)
        return (probas >= threshold).astype(int)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_scaled = self.preprocess_data(X, is_training=False)
        dtest = xgb.DMatrix(X_scaled)
        return self.model.predict(dtest)
    
    def plot_feature_importance(self, feature_names=None):
        """Plot feature importance"""
        importance = self.model.get_score(importance_type='gain')
        if feature_names is not None:
            importance = {feature_names[int(k[1:])]: v for k, v in importance.items()}
        
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame(
            {'Feature': list(importance.keys()), 
             'Importance': list(importance.values())}
        )
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Feature Importance (Gain)')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, X, y):
        """Plot ROC curve"""
        probas = self.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, probas)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        
    def evaluate(self, X, y, threshold=0.5):
        """Evaluate model performance"""
        y_pred = self.predict(X, threshold)
        
        # Print classification report
        print("\nClassification Report:")
        print("----------------------")
        print(classification_report(y, y_pred))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("----------------")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Plot ROC curve
        self.plot_roc_curve(X, y)

def main():
    # Generate sample data (replace with your actual data)
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + X[:, 1] + np.random.randn(1000) > 0).astype(int)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize and train model
    classifier = XGBoostClassifier()
    classifier.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on training set
    print("\nTraining Set Evaluation:")
    print("------------------------")
    classifier.evaluate(X_train, y_train)
    
    # Evaluate on validation set
    print("\nValidation Set Evaluation:")
    print("-------------------------")
    classifier.evaluate(X_val, y_val)
    
    # Evaluate on test set
    print("\nTest Set Evaluation:")
    print("-------------------")
    classifier.evaluate(X_test, y_test)
    
    # Print final summary
    print("\nFinal Model Performance Summary:")
    print("--------------------------------")
    
    # Calculate and display metrics for all sets
    train_pred = classifier.predict(X_train)
    val_pred = classifier.predict(X_val)
    test_pred = classifier.predict(X_test)
    
    train_acc = (train_pred == y_train).mean()
    val_acc = (val_pred == y_val).mean()
    test_acc = (test_pred == y_test).mean()
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Check for potential overfitting
    if train_acc - test_acc > 0.05:  # If training accuracy is significantly higher
        print("\nNote: There might be some overfitting as the training accuracy is")
        print(f"significantly higher than the test accuracy (difference: {train_acc - test_acc:.4f})")
        print("Consider:")
        print("- Adjusting max_depth parameter")
        print("- Increasing min_child_weight")
        print("- Adjusting regularization parameters (lambda, alpha)")
        print("- Using early stopping with more rounds")
        print("- Adding more training data")

    # Plot feature importance
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    classifier.plot_feature_importance(feature_names)

if __name__ == "__main__":
    main()

# # Example usage:
# def main():
#     # Generate sample data (replace with your actual data)
#     np.random.seed(42)
#     X = np.random.randn(1000, 20)
#     y = (X[:, 0] + X[:, 1] + np.random.randn(1000) > 0).astype(int)
    
#     # Split data
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
#     # Initialize and train model
#     classifier = XGBoostClassifier()
#     classifier.train(X_train, y_train, X_val, y_val)
    
    
#     # Evaluate model
#     print("\nTest Set Evaluation:")
#     print("-------------------")
#     classifier.evaluate(X_test, y_test)
    
#     # Plot feature importance
#     feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
#     classifier.plot_feature_importance(feature_names)

# if __name__ == "__main__":
#     main()
