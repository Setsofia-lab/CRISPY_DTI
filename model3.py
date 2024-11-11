import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional

class CustomDataset(Dataset):
    """Custom Dataset for PyTorch DataLoader"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    """Multilayer Perceptron Model"""
    def __init__(self, input_size: int, hidden_layers: List[int], dropout_rate: float = 0.2):
        super(MLP, self).__init__()
        
        # Create list to hold all layers
        layers = []
        
        # Input layer
        prev_size = input_size
        
        # Add hidden layers with dropout
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class MLPClassifier:
    def __init__(self, 
                 input_size: int,
                 hidden_layers: List[int] = [128, 64, 32],
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 dropout_rate: float = 0.2,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.dropout_rate = dropout_rate
        
        # Initialize model
        self.model = MLP(input_size, hidden_layers, dropout_rate).to(device)
        self.scaler = StandardScaler()
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def preprocess_data(self, X, y=None, is_training: bool = True):
        """Preprocess features with standardization"""
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        if y is not None:
            return X_scaled, y
        return X_scaled
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch).squeeze()
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += len(y_batch)
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += len(y_batch)
        
        return total_loss / len(val_loader), correct / total
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              early_stopping_patience: int = 10):
        """Train the model with early stopping"""
        
        # Preprocess data
        X_train_scaled, y_train = self.preprocess_data(X_train, y_train)
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val = self.preprocess_data(X_val, y_val, is_training=False)
            
        # Create data loaders
        train_dataset = CustomDataset(X_train_scaled, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = CustomDataset(X_val_scaled, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            
            if X_val is not None:
                val_loss, val_acc = self.validate(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                # Print progress
                print(f'Epoch {epoch+1}/{epochs}:',
                      f'train_loss={train_loss:.4f},',
                      f'train_acc={train_acc:.4f},',
                      f'val_loss={val_loss:.4f},',
                      f'val_acc={val_acc:.4f}')
                
                # Store history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    # Load best model
                    self.model.load_state_dict(torch.load('best_model.pth'))
                    break
            else:
                print(f'Epoch {epoch+1}/{epochs}:',
                      f'train_loss={train_loss:.4f},',
                      f'train_acc={train_acc:.4f}')
                
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.preprocess_data(X, is_training=False)
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            outputs = self.model(X_tensor).squeeze()
            predictions = (outputs >= threshold).cpu().numpy().astype(int)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        X_scaled = self.preprocess_data(X, is_training=False)
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            probas = self.model(X_tensor).squeeze().cpu().numpy()
        
        return probas
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        if 'val_acc' in self.history and len(self.history['val_acc']) > 0:
            plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5):
        """Evaluate model performance"""
        # Get predictions
        y_pred = self.predict(X, threshold)
        y_proba = self.predict_proba(X)
        
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
        fpr, tpr, _ = roc_curve(y, y_proba)
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

def load_and_prepare_data(train_path: str, test_path: str, target_column: str, 
                         val_size: float = 0.2, random_state: int = 42):
    """
    Load and prepare data from CSV files
    
    Parameters:
    train_path: Path to training CSV file
    test_path: Path to test CSV file
    target_column: Name of the target column
    val_size: Proportion of training data to use for validation
    random_state: Random seed for reproducibility
    """
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separate features and target
    X_train_full = train_data.drop(columns=[target_column])
    y_train_full = train_data[target_column]
    
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=val_size, 
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
        train_path='train_data.csv',
        test_path='test_data.csv',
        target_column='Label'  # Replace with your target column name
    )
    
    # Initialize and train model
    mlp = MLPClassifier(
        input_size=X_train.shape[1],  # Number of features
        hidden_layers=[128, 64, 32],
        learning_rate=0.001,
        batch_size=32,
        dropout_rate=0.2
    )
    
    # Train model
    mlp.train(X_train, y_train, X_val, y_val, epochs=100, early_stopping_patience=10)
    
    # Plot training history
    mlp.plot_training_history()
    
    # Evaluate model
    print("\nTest Set Evaluation:")
    print("-------------------")
    mlp.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
