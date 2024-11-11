import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, AllChem, rdFingerprintGenerator
from rdkit.Chem.rdmolops import PatternFingerprint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generate_molecular_fingerprints(smiles_list):
    """
    Generate different types of molecular fingerprints from SMILES
    """
    # Initialize empty lists for different fingerprint types
    mf_list, maccs_list, apf_list, ttf_list, pf_list = [], [], [], [], []
    
    # Create molecule objects
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    for mol in molecules:
        if mol is not None:
            # Morgan Fingerprints
            morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
            mf = morgan_gen.GetFingerprint(mol)
            mf_arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(mf, mf_arr)
            mf_list.append(mf_arr)
            
            # MACCS Keys
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
            maccs_list.append(maccs_arr)
            
            # Atom Pair Fingerprints
            apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=4096)
            apf = apgen.GetFingerprint(mol)
            apf_arr = np.array(apf)
            apf_list.append(apf_arr)
            
            # Topological Torsion Fingerprints
            ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
            ttf = ttgen.GetFingerprint(mol)
            ttf_arr = np.array(ttf)
            ttf_list.append(ttf_arr)
            
            # Pattern Fingerprints
            pf = PatternFingerprint(mol)
            pf_arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(pf, pf_arr)
            pf_list.append(pf_arr)
        
    return pd.DataFrame(mf_list), pd.DataFrame(maccs_list), pd.DataFrame(apf_list), \
           pd.DataFrame(ttf_list), pd.DataFrame(pf_list)

def encode_protein_sequence(sequence):
    """
    One-hot encode protein sequence
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    
    ohe = np.zeros((len(sequence), len(amino_acids)), dtype=int)
    for i, aa in enumerate(sequence):
        if aa in aa_to_int:
            ohe[i, aa_to_int[aa]] = 1
    return ohe.flatten()

def preprocess_dti_data(df, test_size=0.2, random_state=42):
    """
    Preprocess DTI data and prepare it for XGBoost training
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'SMILES', 'Target Sequence', and 'Label' columns
    test_size : float
        Proportion of dataset to include in the test split
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing train and test data, along with preprocessing objects
    """
    # Generate molecular fingerprints
    mf_df, maccs_df, apf_df, ttf_df, pf_df = generate_molecular_fingerprints(df['SMILES'])
    
    # Encode protein sequences
    protein_embeddings = np.vstack(df['Target Sequence'].apply(encode_protein_sequence))
    protein_df = pd.DataFrame(protein_embeddings)
    
    # Create dictionary of embeddings
    embedding_dict = {
        'protein': protein_df,
        'morgan': mf_df,
        'maccs': maccs_df,
        'atom_pair': apf_df,
        'topological': ttf_df,
        'pattern': pf_df
    }
    
    # Initialize scalers dictionary
    scalers = {}
    
    # Scale each embedding type
    scaled_embeddings = {}
    for name, embedding in embedding_dict.items():
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(embedding)
        scaled_embeddings[name] = pd.DataFrame(scaled_data)
        scalers[name] = scaler
    
    # Combine all scaled embeddings
    X = pd.concat([scaled_embeddings[name] for name in scaled_embeddings.keys()], axis=1)
    y = df['Label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scalers': scalers,
        'feature_names': list(scaled_embeddings.keys())
    }

def prepare_new_data(df, scalers):
    """
    Prepare new data using fitted scalers
    
    Parameters:
    -----------
    df : pandas.DataFrame
        New data to be processed
    scalers : dict
        Dictionary of fitted StandardScaler objects
    
    Returns:
    --------
    pandas.DataFrame : Processed features ready for prediction
    """
    # Generate fingerprints
    mf_df, maccs_df, apf_df, ttf_df, pf_df = generate_molecular_fingerprints(df['SMILES'])
    
    # Encode protein sequences
    protein_embeddings = np.vstack(df['Target Sequence'].apply(encode_protein_sequence))
    protein_df = pd.DataFrame(protein_embeddings)
    
    # Create dictionary of embeddings
    embedding_dict = {
        'protein': protein_df,
        'morgan': mf_df,
        'maccs': maccs_df,
        'atom_pair': apf_df,
        'topological': ttf_df,
        'pattern': pf_df
    }
    
    # Scale each embedding type using provided scalers
    scaled_embeddings = {}
    for name, embedding in embedding_dict.items():
        scaled_data = scalers[name].transform(embedding)
        scaled_embeddings[name] = pd.DataFrame(scaled_data)
    
    # Combine all scaled embeddings
    df_embeddings = pd.concat([scaled_embeddings[name] for name in scaled_embeddings.keys()], axis=1)
    
    return df_embeddings

# Example usage:
# def main():
#     # Load your data
#     df = pd.read_csv('BIOSNAP.csv')
    
#     # Preprocess data
#     processed_data = preprocess_dti_data(df)
    
#     # XGBoost parameters
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
    
#     # Initialize and train model
#     model = XGBoostModel(params, random_state=42)
#     results = model.fit(
#         processed_data['X_train'], 
#         processed_data['y_train'],
#         SquaredErrorObjective(),
#         num_boost_round=100,
#         verbose=True
#     )
    
#     # Make predictions
#     y_pred = model.predict(processed_data['X_test'])
    
#     # Calculate metrics
#     test_loss = SquaredErrorObjective().loss(processed_data['y_test'], y_pred)
#     print(f'Test Loss: {test_loss:.4f}')

# if __name__ == "__main__":
#     main()