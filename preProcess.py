import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, AllChem, rdFingerprintGenerator
from rdkit.Chem.rdmolops import PatternFingerprint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generate_labeled_fingerprints(smiles_list):
    fingerprints = {
        'morgan': {
            'generator': rdFingerprintGenerator.GetMorganGenerator(radius=2),
            'size': 2048,
            'prefix': 'MORGAN'
        },
        'maccs': {
            'generator': None,
            'size': 166,
            'prefix': 'MACCS'
        },
        'atom_pair': {
            'generator': rdFingerprintGenerator.GetAtomPairGenerator(fpSize=4096),
            'size': 4096,
            'prefix': 'ATOMPAIR'
        },
        'topological': {
            'generator': rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048),
            'size': 2048,
            'prefix': 'TOPO'
        },
        'pattern': {
            'generator': None,
            'size': 2048,
            'prefix': 'PATTERN'
        }
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
        fp_df.columns = [f'{fp_info["prefix"]}_{i}' for i in range(fp_df.shape[1])]
        results[fp_type] = fp_df
    
    return results

def encode_protein_sequence(sequence, max_length=1000):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    n_amino_acids = len(amino_acids)
    
    sequence = sequence[:max_length] if len(sequence) > max_length else sequence
    ohe = np.zeros((max_length, n_amino_acids), dtype=int)
    
    for i, aa in enumerate(sequence):
        if aa in aa_to_int:
            ohe[i, aa_to_int[aa]] = 1
            
    return ohe.flatten()

def process_dti_data_with_labels(df, max_seq_length=None):
    fp_dict = generate_labeled_fingerprints(df['SMILES'])
    
    if max_seq_length is None:
        max_seq_length = df['Target Sequence'].str.len().max()
    
    protein_embeddings = np.vstack([
        encode_protein_sequence(seq, max_seq_length) 
        for seq in df['Target Sequence']
    ])
    
    protein_df = pd.DataFrame(protein_embeddings)
    protein_df.columns = [f'PROTEIN_{i}' for i in range(protein_df.shape[1])]
    fp_dict['protein'] = protein_df
    
    scalers = {}
    for name, embedding in fp_dict.items():
        scaler = StandardScaler()
        fp_dict[name] = pd.DataFrame(
            scaler.fit_transform(embedding),
            columns=embedding.columns
        )
        scalers[name] = scaler
    
    return fp_dict, scalers

def prepare_split_dti_data(input_file, test_size=0.2, random_state=42, max_seq_length=None):
    df = pd.read_csv(input_file)
    feature_dict, scalers = process_dti_data_with_labels(df, max_seq_length=max_seq_length)
    
    X = pd.concat(feature_dict.values(), axis=1)
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    feature_groups = {
        'morgan': X.filter(like='MORGAN_').columns.tolist(),
        'maccs': X.filter(like='MACCS_').columns.tolist(),
        'atom_pair': X.filter(like='ATOMPAIR_').columns.tolist(),
        'topological': X.filter(like='TOPO_').columns.tolist(),
        'pattern': X.filter(like='PATTERN_').columns.tolist(),
        'protein': X.filter(like='PROTEIN_').columns.tolist()
    }
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scalers': scalers,
        'feature_groups': feature_groups,
        'all_features': X.columns.tolist()
    }

def visualize_and_reduce_features(data, corr_threshold=0.95):
    X_train = data['X_train']
    corr_matrix = X_train.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
    plt.title("Correlation Matrix of Features")
    plt.show()
    
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                to_drop.add(corr_matrix.columns[i])
    
    print(f"Features to drop based on threshold {corr_threshold}: {to_drop}")
    
    X_train_reduced = X_train.drop(columns=to_drop)
    X_test_reduced = data['X_test'].drop(columns=to_drop)
    
    X_train_reduced.to_csv('train_data_reduced.csv', index=False)
    X_test_reduced.to_csv('test_data_reduced.csv', index=False)
    
    return {
        'X_train_reduced': X_train_reduced,
        'X_test_reduced': X_test_reduced,
        'y_train': data['y_train'],
        'y_test': data['y_test']
    }

if __name__ == "__main__":
    data = prepare_split_dti_data('/Users/samuelsetsofia/dev/projects/DTI_Crispy/BIOSNAP.csv')
    reduced_data = visualize_and_reduce_features(data)
    
    train_data_reduced = pd.concat([reduced_data['X_train_reduced'], reduced_data['y_train']], axis=1)
    test_data_reduced = pd.concat([reduced_data['X_test_reduced'], reduced_data['y_test']], axis=1)
    
    train_data_reduced.to_csv('train_data_reduced_with_labels.csv', index=False)
    test_data_reduced.to_csv('test_data_reduced_with_labels.csv', index=False)
    
    feature_groups_df = pd.DataFrame([{
        'group': group,
        'features': ','.join(features)
    } for group, features in data['feature_groups'].items()])
    feature_groups_df.to_csv('feature_groups.csv', index=False)
