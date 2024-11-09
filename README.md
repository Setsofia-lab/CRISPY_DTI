# Drug-Target Interaction (DTI) Prediction

This project focuses on predicting drug-target interactions (DTI) using machine learning techniques. The dataset includes protein sequences, drug SMILES strings, and various molecular fingerprints (MACCS Keys, Morgan, AP, and Topological Fusion). The goal is to predict whether a drug will bind to a target protein based on these features.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [Dependencies](#dependencies)
- [Results](#results)
- [License](#license)

## Project Overview
Drug-target interactions (DTIs) are crucial in understanding how drugs affect biological systems. This project aims to predict the interactions between drugs and proteins based on their molecular features. The dataset includes molecular fingerprints for drugs and proteins, as well as one-hot encoded protein sequences.

### Main Objectives:
- Perform feature extraction on drug molecules and target proteins.
- Use machine learning models (e.g., XGBoost) to predict drug-target interactions.
- Evaluate model performance using various metrics.

## Data
The dataset contains the following features:
1. **Protein Sequences**: One-hot encoded representation of protein sequences.
2. **Drug SMILES**: SMILES strings representing drug molecules.
3. **Fingerprints**: 
   - MACCS Keys
   - Morgan Fingerprints
   - AP Fingerprints
   - Topological Fusion Fingerprints
   
All features are stored in separate data frames and will be concatenated before model training.

## Installation
To run the project, you need to set up your Python environment and install the necessary dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/Setsofia-lab/CRISPY_DTI.git
   cd CRISPY_DTI
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv rdkitenv
   source rdkitenv/bin/activate  # On Windows, use rdkitenv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preprocessing**: 
   - The dataset should be prepared and cleaned before training the model. The data includes features like protein sequences, drug SMILES, and various molecular fingerprints.
   
2. **Training**: 
   - After preprocessing the data, the `model.py` script can be used to train the model using XGBoost or other machine learning models.
   - The model training involves splitting the data into training, validation, and test sets, followed by training the model with the chosen features.

3. **Prediction**:
   - After training the model, use the trained model to predict drug-target interactions on new or unseen data.

### Example:
```bash
python model.py
```

## Modeling Approach
- **Feature Engineering**: 
  - Protein sequences are one-hot encoded.
  - Drug molecules are represented by molecular fingerprints such as MACCS Keys, Morgan, AP, and Topological Fusion fingerprints.
  
- **Model**:
  - The model is built using XGBoost, which is a powerful gradient boosting algorithm known for its high performance and ability to handle high-dimensional data effectively.
  
- **Evaluation**:
  - The model is evaluated using accuracy, precision, recall, F1-score, and other relevant metrics.

## Dependencies
This project requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `rdkit` (for processing SMILES strings)
- `matplotlib` (for data visualization)

To install the dependencies, use:
```bash
pip install -r requirements.txt
```

## Results
- After training the model, evaluate its performance on a test dataset.
- Model performance can be assessed using metrics such as accuracy, ROC-AUC, and confusion matrix.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Key Points in the Markdown:
- **Project Overview**: Describes the goal of predicting drug-target interactions using molecular features.
- **Data Section**: Details on the data used, including protein sequences and drug SMILES.
- **Installation and Usage**: Step-by-step guide to set up the project and run it.
- **Modeling Approach**: Briefly explains the choice of model and evaluation metrics.
- **Dependencies**: Lists necessary libraries for the project.

