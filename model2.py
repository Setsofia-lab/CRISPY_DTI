import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from preProcess import df_embeddings

class SquaredErrorObjective():
    def loss(self, y, pred): return np.mean((y - pred)**2)
    def gradient(self, y, pred): return pred - y
    def hessian(self, y, pred): return np.ones(len(y))

def prepare_embeddings_for_xgboost(df_embeddings):
    """
    Prepare the concatenated embeddings for XGBoost training
    
    Parameters:
    -----------
    df_embeddings : pandas.DataFrame
        DataFrame containing protein and molecular fingerprint embeddings with labels
    
    Returns:
    --------
    dict : Dictionary containing processed train/test data and scalers
    """
    # Separate features and labels
    X = df_embeddings.drop('Label', axis=1)
    y = df_embeddings['Label']
    
    # Split embeddings by type
    feature_sizes = {
        'protein': len(X.filter(like='Target').columns),
        'MACCS': len(X.filter(like='MACCS').columns),
        'MF': len(X.filter(like='MF').columns),
        'TFF': len(X.filter(like='TFF').columns),
        'PF': len(X.filter(like='PF').columns),
        'APF': len(X.filter(like='APF').columns)
    }
    
    # Initialize scalers dictionary
    scalers = {}
    scaled_features = []
    current_pos = 0
    
    # Scale each embedding type separately
    for name, size in feature_sizes.items():
        if size > 0:
            features = X.iloc[:, current_pos:current_pos + size]
            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)
            scaled_features.append(pd.DataFrame(scaled))
            scalers[name] = scaler
            current_pos += size
    
    # Combine scaled features
    X_scaled = pd.concat(scaled_features, axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scalers': scalers,
        'feature_sizes': feature_sizes
    }

class XGBoostModel():
    def __init__(self, params, random_seed=None):
        self.params = defaultdict(lambda: None, params)
        self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0
        self.learning_rate = self.params['learning_rate'] if self.params['learning_rate'] else 0.3
        self.base_prediction = self.params['base_score'] if self.params['base_score'] else 0.5
        self.max_depth = self.params['max_depth'] if self.params['max_depth'] else 5
        self.rng = np.random.default_rng(seed=random_seed)
    
    def fit(self, X, y, objective, num_boost_round, verbose=False):
        result_array = [[0 for _ in range(num_boost_round)] for _ in range(2)]
        current_predictions = self.base_prediction * np.ones(shape=y.shape)
        self.boosters = []
        
        for i in range(num_boost_round):
            gradients = objective.gradient(y, current_predictions)
            hessians = objective.hessian(y, current_predictions)
            
            sample_idxs = None if self.subsample == 1.0 \
                else self.rng.choice(len(y), size=int(self.subsample*len(y)), replace=False)
            
            booster = TreeBooster(X, gradients, hessians, 
                                self.params, self.max_depth, sample_idxs)
            current_predictions += self.learning_rate * booster.predict(X)
            self.boosters.append(booster)
            
            prediction_now = objective.loss(y, current_predictions)
            result_array[0][i] = prediction_now
            result_array[1][i] = gradients.iloc[0] if isinstance(gradients, pd.Series) else gradients[0]
            
            if verbose and i % 10 == 0: 
                print(f'[{i}] train loss = {prediction_now:.6f}')
        
        return result_array
    
    def predict(self, X):
        return (self.base_prediction + self.learning_rate 
                * np.sum([booster.predict(X) for booster in self.boosters], axis=0))

class TreeBooster():
    def __init__(self, X, g, h, params, max_depth, idxs=None):
        self.params = params
        self.max_depth = max_depth
        self.min_child_weight = params['min_child_weight'] if params['min_child_weight'] else 1.0
        self.reg_lambda = params['reg_lambda'] if params['reg_lambda'] else 1.0
        self.gamma = params['gamma'] if params['gamma'] else 0.0
        self.colsample_bynode = params['colsample_bynode'] if params['colsample_bynode'] else 1.0
        
        if isinstance(g, pd.Series): g = g.values
        if isinstance(h, pd.Series): h = h.values
        if idxs is None: idxs = np.arange(len(g))
        
        self.X, self.g, self.h, self.idxs = X, g, h, idxs
        self.n, self.c = len(idxs), X.shape[1]
        self.value = -g[idxs].sum() / (h[idxs].sum() + self.reg_lambda)
        self.best_score_so_far = 0.
        
        if self.max_depth > 0:
            self._maybe_insert_child_nodes()
    
    def _maybe_insert_child_nodes(self):
        for i in range(self.c): 
            self._find_better_split(i)
        if self.is_leaf: 
            return
        
        x = self.X.values[self.idxs, self.split_feature_idx]
        left_idx = np.nonzero(x <= self.threshold)[0]
        right_idx = np.nonzero(x > self.threshold)[0]
        
        self.left = TreeBooster(self.X, self.g, self.h, self.params, 
                              self.max_depth - 1, self.idxs[left_idx])
        self.right = TreeBooster(self.X, self.g, self.h, self.params, 
                               self.max_depth - 1, self.idxs[right_idx])
    
    @property
    def is_leaf(self): 
        return self.best_score_so_far == 0.
    
    def _find_better_split(self, feature_idx):
        x = self.X.values[self.idxs, feature_idx]
        g, h = self.g[self.idxs], self.h[self.idxs]
        sort_idx = np.argsort(x)
        sort_g, sort_h, sort_x = g[sort_idx], h[sort_idx], x[sort_idx]
        sum_g, sum_h = g.sum(), h.sum()
        sum_g_right, sum_h_right = sum_g, sum_h
        sum_g_left, sum_h_left = 0., 0.

        for i in range(0, self.n - 1):
            g_i, h_i = sort_g[i], sort_h[i]
            x_i, x_i_next = sort_x[i], sort_x[i + 1]
            
            sum_g_left += g_i
            sum_g_right -= g_i
            sum_h_left += h_i
            sum_h_right -= h_i
            
            if sum_h_left < self.min_child_weight or x_i == x_i_next:
                continue
            if sum_h_right < self.min_child_weight:
                break

            gain = 0.5 * ((sum_g_left**2 / (sum_h_left + self.reg_lambda))
                         + (sum_g_right**2 / (sum_h_right + self.reg_lambda))
                         - (sum_g**2 / (sum_h + self.reg_lambda))
                         ) - self.gamma/2
            
            if gain > self.best_score_so_far:
                self.split_feature_idx = feature_idx
                self.best_score_so_far = gain
                self.threshold = (x_i + x_i_next) / 2
    
    def predict(self, X):
        return np.array([self._predict_row(row) for _, row in X.iterrows()])

    def _predict_row(self, row):
        if self.is_leaf:
            return self.value
        child = self.left if row[self.split_feature_idx] <= self.threshold else self.right
        return child._predict_row(row)
    
    def plot_confusion_matrix(y_true, y_pred, title):
        """
        Plot confusion matrix using seaborn
        """
        cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def calculate_metrics(y_true, y_pred):
        """
        Calculate classification metrics
        """
        y_pred_binary = (y_pred > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average='binary')
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train_dti_model(df_embeddings, params=None, num_boost_round=100, verbose=True):
        """
        Train XGBoost model on DTI embeddings with additional evaluation metrics
        
        Parameters:
        -----------
        df_embeddings : pandas.DataFrame
            DataFrame containing concatenated embeddings and labels
        params : dict
            XGBoost parameters
        num_boost_round : int
            Number of boosting rounds
        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        dict : Dictionary containing model, results, and evaluation metrics
        """
        if params is None:
            params = {
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'reg_lambda': 1.5,
                'gamma': 0.1,
                'min_child_weight': 25,
                'base_score': 0.0,
            }
        
        # Prepare data
        processed_data = prepare_embeddings_for_xgboost(df_embeddings)
        
        # Initialize and train model
        model = XGBoostModel(params, random_state=42)
        results = model.fit(
            processed_data['X_train'],
            processed_data['y_train'],
            SquaredErrorObjective(),
            num_boost_round,
            verbose
        )
        
        # Make predictions
        train_pred = model.predict(processed_data['X_train'])
        test_pred = model.predict(processed_data['X_test'])
        
        # Calculate metrics
        train_loss = SquaredErrorObjective().loss(processed_data['y_train'], train_pred)
        test_loss = SquaredErrorObjective().loss(processed_data['y_test'], test_pred)
        
        # Calculate classification metrics
        train_metrics = calculate_metrics(processed_data['y_train'], train_pred)
        test_metrics = calculate_metrics(processed_data['y_test'], test_pred)
        
        # Plot training progress
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(results[0])
        plt.xlabel('Boost round')
        plt.ylabel('Train loss')
        plt.title('Training Loss Progress')
        
        plt.subplot(1, 2, 2)
        plt.plot(results[1])
        plt.xlabel('Boost round')
        plt.ylabel('Gradient')
        plt.title('Gradient Evolution')
        
        plt.tight_layout()
        plt.show()
        
        # Plot confusion matrices
        plot_confusion_matrix(processed_data['y_train'], train_pred, 'Training Set')
        plot_confusion_matrix(processed_data['y_test'], test_pred, 'Test Set')
        
        return {
            'model': model,
            'results': results,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'processed_data': processed_data
        }

    # Example usage
    if __name__ == "__main__":
        
        # Train model
        results = train_dti_model(df_embeddings)
        
        print("\nFinal Results:")
        print(f"Train Loss: {results['train_loss']:.6f}")
        print(f"Test Loss: {results['test_loss']:.6f}")
        print("\nTraining Set Metrics:")
        print(f"Precision: {results['train_metrics']['precision']:.4f}")
        print(f"Recall: {results['train_metrics']['recall']:.4f}")
        print(f"F1 Score: {results['train_metrics']['f1']:.4f}")
        print("\nTest Set Metrics:")
        print(f"Precision: {results['test_metrics']['precision']:.4f}")
        print(f"Recall: {results['test_metrics']['recall']:.4f}")
        print(f"F1 Score: {results['test_metrics']['f1']:.4f}")

# def train_dti_model(df_embeddings, params=None, num_boost_round=100, verbose=True):
#     """
#     Train XGBoost model on DTI embeddings
    
#     Parameters:
#     -----------
#     df_embeddings : pandas.DataFrame
#         DataFrame containing concatenated embeddings and labels
#     params : dict
#         XGBoost parameters
#     num_boost_round : int
#         Number of boosting rounds
#     verbose : bool
#         Whether to print training progress
        
#     Returns:
#     --------
#     dict : Dictionary containing model, results, and evaluation metrics
#     """
#     if params is None:
#         params = {
#             'learning_rate': 0.1,
#             'max_depth': 6,
#             'subsample': 0.8,
#             'reg_lambda': 1.5,
#             'gamma': 0.1,
#             'min_child_weight': 25,
#             'base_score': 0.0,
#         }
    
#     # Prepare data
#     processed_data = prepare_embeddings_for_xgboost(df_embeddings)
    
#     # Initialize and train model
#     model = XGBoostModel(params, random_state=42)
#     results = model.fit(
#         processed_data['X_train'],
#         processed_data['y_train'],
#         SquaredErrorObjective(),
#         num_boost_round,
#         verbose
#     )
    
#     # Make predictions
#     train_pred = model.predict(processed_data['X_train'])
#     test_pred = model.predict(processed_data['X_test'])
    
#     # Calculate metrics
#     train_loss = SquaredErrorObjective().loss(processed_data['y_train'], train_pred)
#     test_loss = SquaredErrorObjective().loss(processed_data['y_test'], test_pred)
    
#     # Plot results
#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(results[0])
#     plt.xlabel('Boost round')
#     plt.ylabel('Train loss')
#     plt.title('Training Loss Progress')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(results[1])
#     plt.xlabel('Boost round')
#     plt.ylabel('Gradient')
#     plt.title('Gradient Evolution')
    
#     plt.tight_layout()
#     plt.show()
    
#     return {
#         'model': model,
#         'results': results,
#         'train_loss': train_loss,
#         'test_loss': test_loss,
#         'processed_data': processed_data
#     }

# # Example usage
# if __name__ == "__main__":
#     # Assuming df_embeddings is your concatenated embeddings DataFrame
#     # df_embeddings = pd.concat([protein_embedding, MACCS_embeddings, MF_embedding, 
#     #                           TFF_embedding, PF_embeddings, APF_embedding, labels], axis=1)
    
#     # Train model
#     results = train_dti_model(df_embeddings)
    
#     print(f"\nFinal Results:")
#     print(f"Train Loss: {results['train_loss']:.6f}")
#     print(f"Test Loss: {results['test_loss']:.6f}")



# import math
# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# # import df_embeddings

# class DimensionalityReducer:
#     def __init__(self, n_components=50):
#         self.n_components = n_components
#         self.scalers = {}
#         self.pcas = {}
        
#     def fit_transform(self, embeddings_dict):
#         """
#         # Fit and transform each embedding type separately
#         """
#         reduced_embeddings = {}
#         for name, embedding in embeddings_dict.items():
#             # Standardize
#             self.scalers[name] = StandardScaler()
#             scaled_data = self.scalers[name].fit_transform(embedding)
            
#             # Apply PCA
#             self.pcas[name] = PCA(n_components=min(self.n_components, embedding.shape[1]))
#             reduced_data = self.pcas[name].fit_transform(scaled_data)
#             reduced_embeddings[name] = reduced_data
            
#             variance_explained = np.sum(self.pcas[name].explained_variance_ratio_)
#             print(f"{name} - Variance explained: {variance_explained:.3f}")
            
#         return reduced_embeddings
    
#     def transform(self, embeddings_dict):
#         """
#         Transform new data using fitted scalers and PCAs
#         """
#         reduced_embeddings = {}
#         for name, embedding in embeddings_dict.items():
#             scaled_data = self.scalers[name].transform(embedding)
#             reduced_data = self.pcas[name].transform(scaled_data)
#             reduced_embeddings[name] = reduced_data
#         return reduced_embeddings

# class XGBoostModel():
#     '''XGBoost from Scratch with support for DTI prediction
#     '''
#     def __init__(self, params, random_seed=None):
#         self.params = defaultdict(lambda: None, params)
#         self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0
#         self.learning_rate = self.params['learning_rate'] if self.params['learning_rate'] else 0.3
#         self.base_prediction = self.params['base_score'] if self.params['base_score'] else 0.5
#         self.max_depth = self.params['max_depth'] if self.params['max_depth'] else 5
#         self.rng = np.random.default_rng(seed=random_seed)
        
#     def fit(self, X, y, objective, num_boost_round, verbose=False):
#         result_array = [[0 for _ in range(num_boost_round)] for _ in range(2)]
        
#         current_predictions = self.base_prediction * np.ones(shape=y.shape)
#         self.boosters = []
        
#         for i in range(num_boost_round):
#             gradients = objective.gradient(y, current_predictions)
#             hessians = objective.hessian(y, current_predictions)
            
#             sample_idxs = None if self.subsample == 1.0 \
#                 else self.rng.choice(len(y), 
#                                    size=math.floor(self.subsample*len(y)), 
#                                    replace=False)
            
#             booster = TreeBooster(X, gradients, hessians, 
#                                 self.params, self.max_depth, sample_idxs)
#             current_predictions += self.learning_rate * booster.predict(X)
#             self.boosters.append(booster)
            
#             prediction_now = objective.loss(y, current_predictions)
#             result_array[0][i] = prediction_now
#             result_array[1][i] = gradients.iloc[0]
            
#             if verbose and i % 10 == 0: 
#                 print(f'[{i}] train loss = {prediction_now}')
                
#         return result_array
    
#     def predict(self, X):
#         return (self.base_prediction + self.learning_rate 
#                 * np.sum([booster.predict(X) for booster in self.boosters], axis=0))

# # Rest of the TreeBooster class remains the same as in your original code
# # [Previous TreeBooster class code here]

# def prepare_dti_data(df_embeddings):
#     """
#     Prepare DTI embeddings for modeling
#     """
#     # Split embeddings into features and labels
#     X = df_embeddings.drop('labels', axis=1)
#     y = df_embeddings['labels']
    
#     # Create dictionary of embeddings for dimensionality reduction
#     embedding_dict = {
#         'protein': X.filter(like='protein'),
#         'MACCS': X.filter(like='MACCS'),
#         'MF': X.filter(like='MF'),
#         'TFF': X.filter(like='TFF'),
#         'PF': X.filter(like='PF'),
#         'APF': X.filter(like='APF')
#     }
    
#     return embedding_dict, y

# def main():
#     # Load your embeddings
#     # df_embeddings should already be created as mentioned in your concatenation
    
#     # Prepare data
#     embedding_dict, labels = prepare_dti_data(df_embeddings)
    
#     # Initialize dimensionality reducer
#     reducer = DimensionalityReducer(n_components=50)
    
#     # Reduce dimensionality
#     reduced_embeddings = reducer.fit_transform(embedding_dict)
    
#     # Combine reduced embeddings
#     X_reduced = np.hstack([reduced_embeddings[name] for name in reduced_embeddings.keys()])
#     X_reduced = pd.DataFrame(X_reduced)
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_reduced, labels, test_size=0.2, random_state=42, stratify=labels
#     )
    
#     # Define XGBoost parameters
#     params = {
#         'learning_rate': 0.1,
#         'max_depth': 6,
#         'subsample': 0.8,
#         'reg_lambda': 1.5,
#         'gamma': 0.1,
#         'min_child_weight': 25,
#         'base_score': 0.0,
#         'tree_method': 'exact',
#     }
    
#     num_boost_round = 100
    
#     # Train model
#     model = XGBoostModel(params, random_seed=42)
#     results = model.fit(X_train, y_train, SquaredErrorObjective(), num_boost_round, verbose=True)
    
#     # Make predictions
#     pred_train = model.predict(X_train)
#     pred_test = model.predict(X_test)
    
#     # Calculate metrics
#     train_loss = SquaredErrorObjective().loss(y_train, pred_train)
#     test_loss = SquaredErrorObjective().loss(y_test, pred_test)
    
#     print(f'Train loss: {train_loss:.4f}')
#     print(f'Test loss: {test_loss:.4f}')
    
#     # Plot training progress
#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(results[0])
#     plt.xlabel('Boost round')
#     plt.ylabel('Train loss')
#     plt.title('Training Loss Progress')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(results[1])
#     plt.xlabel('Boost round')
#     plt.ylabel('Gradient')
#     plt.title('Gradient Evolution')
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()