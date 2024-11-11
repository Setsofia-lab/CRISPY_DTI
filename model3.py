import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SquaredErrorObjective:
    def loss(self, y, pred): 
        return np.mean((y - pred)**2)
    
    def gradient(self, y, pred): 
        return pred - y
    
    def hessian(self, y, pred): 
        return np.ones(len(y))

class TreeNode:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.score = None
    
    def predict(self, x):
        if self.is_leaf:
            return self.score
        
        if x[self.feature_idx] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

class TreeBooster:
    def __init__(self, max_depth, gamma, min_child_weight, subsample=1.0):
        self.max_depth = max_depth
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.root = None
    
    def find_best_split(self, X, g, h, row_indices):
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        
        n_features = X.shape[1]
        
        G = np.sum(g[row_indices])
        H = np.sum(h[row_indices])
        current_score = -(G * G) / (H + self.gamma)
        
        for feature in range(n_features):
            feature_values = X[row_indices, feature]
            sorted_indices = np.argsort(feature_values)
            sorted_feature_values = feature_values[sorted_indices]
            
            unique_values = np.unique(sorted_feature_values)
            if len(unique_values) == 1:
                continue
            
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                left_indices = row_indices[left_mask]
                right_indices = row_indices[right_mask]
                
                if len(left_indices) < self.min_child_weight or len(right_indices) < self.min_child_weight:
                    continue
                
                GL = np.sum(g[left_indices])
                HL = np.sum(h[left_indices])
                GR = np.sum(g[right_indices])
                HR = np.sum(h[right_indices])
                
                gain = (GL * GL) / (HL + self.gamma) + (GR * GR) / (HR + self.gamma) - current_score
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices
        
        return best_gain, best_feature, best_threshold, best_left_indices, best_right_indices
    
    def build_tree(self, X, g, h, row_indices, depth=0):
        node = TreeNode()
        
        G = np.sum(g[row_indices])
        H = np.sum(h[row_indices])
        
        if depth == self.max_depth or len(row_indices) < self.min_child_weight:
            node.is_leaf = True
            node.score = -G / (H + self.gamma)
            return node
        
        gain, feature, threshold, left_indices, right_indices = self.find_best_split(X, g, h, row_indices)
        
        if gain < self.gamma or feature is None:
            node.is_leaf = True
            node.score = -G / (H + self.gamma)
            return node
        
        node.feature_idx = feature
        node.threshold = threshold
        node.left = self.build_tree(X, g, h, left_indices, depth + 1)
        node.right = self.build_tree(X, g, h, right_indices, depth + 1)
        
        return node
    
    def fit(self, X, g, h):
        n_samples = X.shape[0]
        if self.subsample < 1.0:
            n_subsample = int(n_samples * self.subsample)
            row_indices = np.random.choice(n_samples, n_subsample, replace=False)
        else:
            row_indices = np.arange(n_samples)
        
        self.root = self.build_tree(X, g, h, row_indices)
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self.root.predict(x) for x in X])

class XGBoostModel:
    def __init__(self, params, random_seed=None):
        self.learning_rate = params.get('learning_rate', 0.1)
        self.max_depth = params.get('max_depth', 6)
        self.subsample = params.get('subsample', 1.0)
        self.reg_lambda = params.get('reg_lambda', 1.0)
        self.gamma = params.get('gamma', 0.0)
        self.min_child_weight = params.get('min_child_weight', 1)
        self.base_score = params.get('base_score', 0.0)
        self.trees = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def fit(self, X, y, objective, num_boost_round=100, verbose=False):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        pred = np.full(len(y), self.base_score)
        train_loss_history = []
        
        for boost_round in range(num_boost_round):
            grad = objective.gradient(y, pred)
            hess = objective.hessian(y, pred)
            
            tree = TreeBooster(
                max_depth=self.max_depth,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample
            )
            tree.fit(X, grad, hess)
            
            update = tree.predict(X)
            pred += self.learning_rate * update
            self.trees.append(tree)
            
            loss = objective.loss(y, pred)
            train_loss_history.append(loss)
            
            if verbose and (boost_round + 1) % 10 == 0:
                print(f"Boost round {boost_round + 1}, train loss: {loss:.6f}")
        
        return train_loss_history
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        pred = np.full(len(X), self.base_score)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return 1 / (1 + np.exp(-pred))  # Sigmoid for binary classification
    
def calculate_metrics(y_true, y_pred):
    """Calculate basic classification metrics"""
    y_pred_binary = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix_with_metrics(y_true, y_pred, title):
    """Plot confusion matrix with metrics"""
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Create confusion matrix
    y_pred_binary = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Create figure with subplot
    fig = plt.figure(figsize=(12, 6))
    
    # Create grid specification for layout
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    
    # Confusion matrix subplot
    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax0)
    ax0.set_title(f'Confusion Matrix - {title}')
    ax0.set_ylabel('True Label')
    ax0.set_xlabel('Predicted Label')
    
    # Metrics subplot
    ax1 = fig.add_subplot(gs[1])
    metrics_data = pd.DataFrame(
        list(metrics.values()),
        index=list(metrics.keys()),
        columns=['Score']
    )
    
    # Create a bar plot for metrics
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
    bars = ax1.barh(
        range(len(metrics)),
        metrics_data['Score'],
        color=colors
    )
    
    # Customize metrics visualization
    ax1.set_yticks(range(len(metrics)))
    ax1.set_yticklabels([m.capitalize() for m in metrics.keys()])
    ax1.set_xlim(0, 1)
    ax1.set_title('Performance Metrics')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax1.text(
            width + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            va='center'
        )
    
    plt.tight_layout()
    plt.show()
    
    return metrics

# Update the train_and_evaluate function to use the new plotting function
def train_and_evaluate(train_data_path, test_data_path, val_size=0.2):
    """Train and evaluate XGBoost model"""
    # Load data
    df_train = pd.read_csv(train_data_path)
    X = df_train.drop('Label', axis=1)
    y = df_train['Label']
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Model parameters
    params = {
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'reg_lambda': 1.5,
        'gamma': 0.1,
        'min_child_weight': 25,
        'base_score': 0.0,
    }
    
    # Train model
    model = XGBoostModel(params, random_seed=42)
    loss_history = model.fit(
        X_train, y_train,
        SquaredErrorObjective(),
        num_boost_round=100,
        verbose=True
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Boost Round')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Boosting Rounds')
    plt.show()
    
    # Make predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Plot confusion matrices with metrics
    train_metrics = plot_confusion_matrix_with_metrics(y_train, train_pred, 'Training Set')
    val_metrics = plot_confusion_matrix_with_metrics(y_val, val_pred, 'Validation Set')
    
    # Evaluate on test set
    df_test = pd.read_csv(test_data_path)
    X_test = df_test.drop('Label', axis=1)
    y_test = df_test['Label']
    
    test_pred = model.predict(X_test)
    test_metrics = plot_confusion_matrix_with_metrics(y_test, test_pred, 'Test Set')
    
    return {
        'model': model,
        'loss_history': loss_history,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        },
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
    }

# def plot_confusion_matrix(y_true, y_pred, title):
#     """Plot confusion matrix"""
#     cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Confusion Matrix - {title}')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.show()

# def calculate_metrics(y_true, y_pred):
#     """Calculate basic classification metrics"""
#     y_pred_binary = (y_pred > 0.5).astype(int)
#     cm = confusion_matrix(y_true, y_pred_binary)
#     tn, fp, fn, tp = cm.ravel()
    
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#     accuracy = (tp + tn) / (tp + tn + fp + fn)
    
#     return {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1
#     }

# def train_and_evaluate(train_data_path, test_data_path, val_size=0.2):
#     """Train and evaluate XGBoost model"""
#     # Load data
#     df_train = pd.read_csv(train_data_path)
#     X = df_train.drop('Label', axis=1)
#     y = df_train['Label']
    
#     # Split into train and validation
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=val_size, random_state=42, stratify=y
#     )
    
#     print(f"Training set shape: {X_train.shape}")
#     print(f"Validation set shape: {X_val.shape}")
    
#     # Model parameters
#     params = {
#         'learning_rate': 0.1,
#         'max_depth': 6,
#         'subsample': 0.8,
#         'reg_lambda': 1.5,
#         'gamma': 0.1,
#         'min_child_weight': 25,
#         'base_score': 0.0,
#     }
    
#     # Train model
#     model = XGBoostModel(params, random_seed=42)
#     loss_history = model.fit(
#         X_train, y_train,
#         SquaredErrorObjective(),
#         num_boost_round=100,
#         verbose=True
#     )
    
#     # Plot training loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(loss_history)
#     plt.xlabel('Boost Round')
#     plt.ylabel('Training Loss')
#     plt.title('Training Loss vs Boosting Rounds')
#     plt.show()
    
#     # Make predictions
#     train_pred = model.predict(X_train)
#     val_pred = model.predict(X_val)
    
#     # Plot training/validation confusion matrices
#     plot_confusion_matrix(y_train, train_pred, 'Training Set')
#     plot_confusion_matrix(y_val, val_pred, 'Validation Set')
    
#     # Calculate metrics
#     train_metrics = calculate_metrics(y_train, train_pred)
#     val_metrics = calculate_metrics(y_val, val_pred)
    
#     print("\nTraining Set Metrics:")
#     for metric, value in train_metrics.items():
#         print(f"{metric.capitalize()}: {value:.4f}")
    
#     print("\nValidation Set Metrics:")
#     for metric, value in val_metrics.items():
#         print(f"{metric.capitalize()}: {value:.4f}")
    
#     # Evaluate on test set
#     df_test = pd.read_csv(test_data_path)
#     X_test = df_test.drop('Label', axis=1)
#     y_test = df_test['Label']
    
#     test_pred = model.predict(X_test)
#     plot_confusion_matrix(y_test, test_pred, 'Test Set')
    
#     test_metrics = calculate_metrics(y_test, test_pred)
#     print("\nTest Set Metrics:")
#     for metric, value in test_metrics.items():
#         print(f"{metric.capitalize()}: {value:.4f}")
    
#     return {
#         'model': model,
#         'loss_history': loss_history,
#         'predictions': {
#             'train': train_pred,
#             'val': val_pred,
#             'test': test_pred
#         },
#         'metrics': {
#             'train': train_metrics,
#             'val': val_metrics,
#             'test': test_metrics
#         }
#     }

# Example usage
if __name__ == "__main__":
    results = train_and_evaluate(
        train_data_path='train_data.csv',
        test_data_path='test_data.csv'
    )