import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, AllChem, rdFingerprintGenerator
from rdkit.Chem.rdmolops import PatternFingerprint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generate_molecular_fingerprints(smiles_list, prefix_names=False):
    """
    Generate different types of molecular fingerprints from SMILES
    
    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
    prefix_names : bool
        Whether to prefix column names with fingerprint type
    
    Returns:
    --------
    dict : Dictionary containing DataFrames for each fingerprint type
    """
    fingerprints = {
        'morgan': {'generator': rdFingerprintGenerator.GetMorganGenerator(radius=2), 'size': 2048},
        'maccs': {'generator': None, 'size': 166},  # MACCS keys handled separately
        'atom_pair': {'generator': rdFingerprintGenerator.GetAtomPairGenerator(fpSize=4096), 'size': 4096},
        'topological': {'generator': rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048), 'size': 2048},
        'pattern': {'generator': None, 'size': 2048}  # Pattern fingerprint handled separately
    }
    
    results = {}
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    valid_mols = [mol for mol in molecules if mol is not None]
    
    for fp_type, fp_info in fingerprints.items():
        fp_arrays = []
        
        for mol in valid_mols:
            if fp_type == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif fp_type == 'pattern':
                fp = PatternFingerprint(mol)
            else:
                fp = fp_info['generator'].GetFingerprint(mol)
            
            fp_arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
            fp_arrays.append(fp_arr)
        
        fp_df = pd.DataFrame(np.vstack(fp_arrays))
        if prefix_names:
            fp_df.columns = [f'{fp_type.upper()}_{i}' for i in range(fp_df.shape[1])]
        results[fp_type] = fp_df
    
    return results

def encode_protein_sequence(sequence, max_length=1000):
    """
    One-hot encode protein sequence with padding or truncation
    
    Parameters:
    -----------
    sequence : str
        Protein sequence to encode
    max_length : int
        Maximum sequence length to consider. Longer sequences will be truncated,
        shorter sequences will be padded with zeros
        
    Returns:
    --------
    numpy.ndarray : Encoded sequence array
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    n_amino_acids = len(amino_acids)
    
    # Truncate or pad sequence
    sequence = sequence[:max_length] if len(sequence) > max_length else sequence
    
    # Initialize zero array
    ohe = np.zeros((max_length, n_amino_acids), dtype=int)
    
    # Fill in the encodings for the actual sequence
    for i, aa in enumerate(sequence):
        if aa in aa_to_int:
            ohe[i, aa_to_int[aa]] = 1
            
    return ohe.flatten()

def process_dti_data(df, prefix_names=False, scalers=None, max_seq_length=None):
    """
    Process DTI data by generating molecular fingerprints and encoding protein sequences
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'SMILES' and 'Target Sequence' columns
    prefix_names : bool
        Whether to prefix column names with feature type
    scalers : dict, optional
        Dictionary of pre-fitted StandardScaler objects
    max_seq_length : int, optional
        Maximum sequence length for protein encoding. If None, will use the length
        of the longest sequence in the dataset
        
    Returns:
    --------
    tuple : (processed_features, scalers)
    """
    # Generate fingerprints
    fp_dict = generate_molecular_fingerprints(df['SMILES'], prefix_names)
    
    # Determine max sequence length if not provided
    if max_seq_length is None:
        max_seq_length = df['Target Sequence'].str.len().max()
    
    # Encode protein sequences with consistent length
    protein_embeddings = np.vstack([
        encode_protein_sequence(seq, max_seq_length) 
        for seq in df['Target Sequence']
    ])
    
    protein_df = pd.DataFrame(protein_embeddings)
    if prefix_names:
        protein_df.columns = [f'TARGET_{i}' for i in range(protein_df.shape[1])]
    fp_dict['protein'] = protein_df
    
    # Scale features if needed
    if scalers is None:
        scalers = {}
        for name, embedding in fp_dict.items():
            scaler = StandardScaler()
            fp_dict[name] = pd.DataFrame(scaler.fit_transform(embedding))
            scalers[name] = scaler
    else:
        for name, embedding in fp_dict.items():
            fp_dict[name] = pd.DataFrame(scalers[name].transform(embedding))
    
    # Combine all features
    processed_features = pd.concat(fp_dict.values(), axis=1)
    
    return processed_features, scalers

def prepare_dti_data(input_file, test_size=0.2, random_state=42, max_seq_length=None):
    """
    Prepare DTI data for ML training
    
    Parameters:
    -----------
    input_file : str
        Path to CSV file containing 'SMILES', 'Target Sequence', and 'Label' columns
    test_size : float
        Proportion of dataset to include in the test split
    random_state : int
        Random state for reproducibility
    max_seq_length : int, optional
        Maximum sequence length for protein encoding
        
    Returns:
    --------
    dict : Dictionary containing train/test splits and preprocessing objects
    """
    df = pd.read_csv(input_file)
    
    # Process features and get scalers
    X, scalers = process_dti_data(df, prefix_names=True, max_seq_length=max_seq_length)
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
        'feature_names': X.columns.tolist()
    }

if __name__ == "__main__":
    # You can specify a maximum sequence length if desired
    data = prepare_dti_data('/Users/samuelsetsofia/dev/projects/DTI_Crispy/BIOSNAP.csv', 
                           max_seq_length=1000)  # adjust this value based on your needs
    
    # Save processed data
    pd.concat([data['X_train'], data['y_train']], axis=1).to_csv('train_data.csv', index=False)
    pd.concat([data['X_test'], data['y_test']], axis=1).to_csv('test_data.csv', index=False)

# def encode_protein_sequence(sequence):
#     """
#     One-hot encode protein sequence
#     """
#     amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
#     aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    
#     ohe = np.zeros((len(sequence), len(amino_acids)), dtype=int)
#     for i, aa in enumerate(sequence):
#         if aa in aa_to_int:
#             ohe[i, aa_to_int[aa]] = 1
#     return ohe.flatten()

# def process_dti_data(df, prefix_names=False, scalers=None):
#     """
#     Process DTI data by generating molecular fingerprints and encoding protein sequences
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         DataFrame containing 'SMILES' and 'Target Sequence' columns
#     prefix_names : bool
#         Whether to prefix column names with feature type
#     scalers : dict, optional
#         Dictionary of pre-fitted StandardScaler objects
    
#     Returns:
#     --------
#     tuple : (processed_features, scalers)
#     """
#     # Generate fingerprints
#     fp_dict = generate_molecular_fingerprints(df['SMILES'], prefix_names)
    
#     # Encode protein sequences
#     protein_embeddings = np.vstack(df['Target Sequence'].apply(encode_protein_sequence))
#     protein_df = pd.DataFrame(protein_embeddings)
#     if prefix_names:
#         protein_df.columns = [f'TARGET_{i}' for i in range(protein_df.shape[1])]
#     fp_dict['protein'] = protein_df
    
#     # Scale features if needed
#     if scalers is None:
#         scalers = {}
#         for name, embedding in fp_dict.items():
#             scaler = StandardScaler()
#             fp_dict[name] = pd.DataFrame(scaler.fit_transform(embedding))
#             scalers[name] = scaler
#     else:
#         for name, embedding in fp_dict.items():
#             fp_dict[name] = pd.DataFrame(scalers[name].transform(embedding))
    
#     # Combine all features
#     processed_features = pd.concat(fp_dict.values(), axis=1)
    
#     return processed_features, scalers

# def prepare_dti_data(input_file, test_size=0.2, random_state=42):
#     """
#     Prepare DTI data for ML training
    
#     Parameters:
#     -----------
#     input_file : str
#         Path to CSV file containing 'SMILES', 'Target Sequence', and 'Label' columns
#     test_size : float
#         Proportion of dataset to include in the test split
#     random_state : int
#         Random state for reproducibility
    
#     Returns:
#     --------
#     dict : Dictionary containing train/test splits and preprocessing objects
#     """
#     df = pd.read_csv(input_file)
    
#     # Process features and get scalers
#     X, scalers = process_dti_data(df, prefix_names=True)
#     y = df['Label']
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )
    
#     return {
#         'X_train': X_train,
#         'X_test': X_test,
#         'y_train': y_train,
#         'y_test': y_test,
#         'scalers': scalers,
#         'feature_names': X.columns.tolist()
#     }

# if __name__ == "__main__":
#     input_file = "your_input_data.csv"  # Replace with your input file path
#     data = prepare_dti_data('/Users/samuelsetsofia/dev/projects/DTI_Crispy/BIOSNAP.csv')
    
#     # Save processed data
#     pd.concat([data['X_train'], data['y_train']], axis=1).to_csv('train_data.csv', index=False)
#     pd.concat([data['X_test'], data['y_test']], axis=1).to_csv('test_data.csv', index=False)




# 
# import numpy as np
# import pandas as pd
# from rdkit import Chem, DataStructs
# from rdkit.Chem import MACCSkeys, AllChem, rdFingerprintGenerator
# from rdkit.Chem.rdmolops import PatternFingerprint
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# def generate_molecular_fingerprints(smiles_list):
#     """
#     Generate different types of molecular fingerprints from SMILES
#     """
#     # Initialize empty lists for different fingerprint types
#     mf_list, maccs_list, apf_list, ttf_list, pf_list = [], [], [], [], []
    
#     # Create molecule objects
#     molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
#     for mol in molecules:
#         if mol is not None:
#             # Morgan Fingerprints
#             morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
#             mf = morgan_gen.GetFingerprint(mol)
#             mf_arr = np.zeros((0,), dtype=np.int8)
#             DataStructs.ConvertToNumpyArray(mf, mf_arr)
#             mf_list.append(mf_arr)
            
#             # MACCS Keys
#             maccs = MACCSkeys.GenMACCSKeys(mol)
#             maccs_arr = np.zeros((0,), dtype=np.int8)
#             DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
#             maccs_list.append(maccs_arr)
            
#             # Atom Pair Fingerprints
#             apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=4096)
#             apf = apgen.GetFingerprint(mol)
#             apf_arr = np.array(apf)
#             apf_list.append(apf_arr)
            
#             # Topological Torsion Fingerprints
#             ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
#             ttf = ttgen.GetFingerprint(mol)
#             ttf_arr = np.array(ttf)
#             ttf_list.append(ttf_arr)
            
#             # Pattern Fingerprints
#             pf = PatternFingerprint(mol)
#             pf_arr = np.zeros((0,), dtype=np.int8)
#             DataStructs.ConvertToNumpyArray(pf, pf_arr)
#             pf_list.append(pf_arr)
        
#     return pd.DataFrame(mf_list), pd.DataFrame(maccs_list), pd.DataFrame(apf_list), \
#            pd.DataFrame(ttf_list), pd.DataFrame(pf_list)

# def encode_protein_sequence(sequence):
#     """
#     One-hot encode protein sequence
#     """
#     amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
#     aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    
#     ohe = np.zeros((len(sequence), len(amino_acids)), dtype=int)
#     for i, aa in enumerate(sequence):
#         if aa in aa_to_int:
#             ohe[i, aa_to_int[aa]] = 1
#     return ohe.flatten()

# def preprocess_dti_data(df, test_size=0.2, random_state=42):
#     """
#     Preprocess DTI data and prepare it for XGBoost training
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         DataFrame containing 'SMILES', 'Target Sequence', and 'Label' columns
#     test_size : float
#         Proportion of dataset to include in the test split
#     random_state : int
#         Random state for reproducibility
    
#     Returns:
#     --------
#     dict : Dictionary containing train and test data, along with preprocessing objects
#     """
#     # Generate molecular fingerprints
#     mf_df, maccs_df, apf_df, ttf_df, pf_df = generate_molecular_fingerprints(df['SMILES'])
    
#     # Encode protein sequences
#     protein_embeddings = np.vstack(df['Target Sequence'].apply(encode_protein_sequence))
#     protein_df = pd.DataFrame(protein_embeddings)
    
#     # Create dictionary of embeddings
#     embedding_dict = {
#         'protein': protein_df,
#         'morgan': mf_df,
#         'maccs': maccs_df,
#         'atom_pair': apf_df,
#         'topological': ttf_df,
#         'pattern': pf_df
#     }
    
#     # Initialize scalers dictionary
#     scalers = {}
    
#     # Scale each embedding type
#     scaled_embeddings = {}
#     for name, embedding in embedding_dict.items():
#         scaler = StandardScaler()
#         scaled_data = scaler.fit_transform(embedding)
#         scaled_embeddings[name] = pd.DataFrame(scaled_data)
#         scalers[name] = scaler
    
#     # Combine all scaled embeddings
#     X = pd.concat([scaled_embeddings[name] for name in scaled_embeddings.keys()], axis=1)
#     y = df['Label']
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )
    
#     return {
#         'X_train': X_train,
#         'X_test': X_test,
#         'y_train': y_train,
#         'y_test': y_test,
#         'scalers': scalers,
#         'feature_names': list(scaled_embeddings.keys())
#     }

# def prepare_new_data(df, scalers):
#     """
#     Prepare new data using fitted scalers
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         New data to be processed
#     scalers : dict
#         Dictionary of fitted StandardScaler objects
    
#     Returns:
#     --------
#     pandas.DataFrame : Processed features ready for prediction
#     """
#     # Generate fingerprints
#     mf_df, maccs_df, apf_df, ttf_df, pf_df = generate_molecular_fingerprints(df['SMILES'])
    
#     # Encode protein sequences
#     protein_embeddings = np.vstack(df['Target Sequence'].apply(encode_protein_sequence))
#     protein_df = pd.DataFrame(protein_embeddings)
    
#     # Create dictionary of embeddings
#     embedding_dict = {
#         'protein': protein_df,
#         'morgan': mf_df,
#         'maccs': maccs_df,
#         'atom_pair': apf_df,
#         'topological': ttf_df,
#         'pattern': pf_df
#     }
    
#     # Scale each embedding type using provided scalers
#     scaled_embeddings = {}
#     for name, embedding in embedding_dict.items():
#         scaled_data = scalers[name].transform(embedding)
#         scaled_embeddings[name] = pd.DataFrame(scaled_data)
    
#     # Combine all scaled embeddings
#     df_embeddings = pd.concat([scaled_embeddings[name] for name in scaled_embeddings.keys()], axis=1)
    
#     return df_embeddings

# def create_embeddings(input_file):
#     """
#     Create embeddings from input data and save them
    
#     Parameters:
#     -----------
#     input_file : str
#         Path to CSV file containing 'SMILES', 'Target Sequence', and 'Label' columns
    
#     Returns:
#     --------
#     pandas.DataFrame : DataFrame containing all embeddings and labels
#     """
#     # Read input data
#     df = pd.read_csv(input_file)
    
#     # Generate molecular fingerprints
#     mf_df, maccs_df, apf_df, ttf_df, pf_df = generate_molecular_fingerprints(df['SMILES'])
    
#     # Rename columns to identify fingerprint types
#     mf_df.columns = [f'MF_{i}' for i in range(mf_df.shape[1])]
#     maccs_df.columns = [f'MACCS_{i}' for i in range(maccs_df.shape[1])]
#     apf_df.columns = [f'APF_{i}' for i in range(apf_df.shape[1])]
#     ttf_df.columns = [f'TFF_{i}' for i in range(ttf_df.shape[1])]
#     pf_df.columns = [f'PF_{i}' for i in range(pf_df.shape[1])]
    
#     # Encode protein sequences
#     protein_embeddings = np.vstack(df['Target Sequence'].apply(encode_protein_sequence))
#     protein_df = pd.DataFrame(protein_embeddings, 
#                             columns=[f'Target_{i}' for i in range(protein_embeddings.shape[1])])
    
#     # Combine all embeddings
#     df_embeddings = pd.concat([
#         protein_df, mf_df, maccs_df, apf_df, ttf_df, pf_df,
#         df['Label']  # Add the labels
#     ], axis=1)
    
#     return df_embeddings

# if __name__ == "__main__":
#     # Create and save embeddings
#     input_file = "your_input_data.csv"  # Replace with your input file path
#     df_embeddings = create_embeddings(input_file)
    
#     # Save embeddings to file
#     df_embeddings.to_csv('preprocessed_embeddings.csv', index=False)

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