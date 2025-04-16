import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Seq import Seq

# Function to calculate peptide length
def calculate_length(sequence):
    return len(sequence)

# Function to calculate net charge at pH 7.0
def calculate_charge(sequence):
    analysis = ProteinAnalysis(sequence)
    return analysis.charge_at_pH(7.0)

# Function to estimate flexibility (approx. by Gly + Pro content)
def estimate_flexibility(sequence):
    gly_pro_content = sequence.count('G') + sequence.count('P')
    return gly_pro_content / len(sequence) if len(sequence) > 0 else 0

# Function to simulate secondary structure prediction (placeholder)
def predict_secondary_structure(sequence):
    return ''.join(['H' if i % 3 == 0 else 'C' for i in range(len(sequence))])

# Function to estimate solvent accessibility (hydrophobicity-based approximation)
def estimate_solvent_accessibility(sequence):
    hydrophilic_scale = {
        'K': 1.8, 'R': 2.5, 'D': 3.0, 'E': 3.0, 'H': 1.0,
        'A': 0.5, 'C': 0.2, 'F': -2.5, 'I': -3.1, 'L': -2.8,
        'M': -1.9, 'N': 0.2, 'P': 0.0, 'Q': 0.2, 'S': 0.3,
        'T': 0.4, 'V': -2.6, 'W': -3.4, 'Y': -2.3, 'G': 0.0
    }
    values = [hydrophilic_scale.get(aa, 0) for aa in sequence]
    return np.mean(values) if values else 0

# Simplified BLOSUM62 scoring
def get_blosum_scores(sequence):
    blosum = {
        ('A', 'A'): 4, ('A', 'C'): 0, ('A', 'D'): -2, ('A', 'E'): -1,
        ('C', 'C'): 9, ('C', 'A'): 0, ('D', 'D'): 4, ('E', 'E'): 5
        # Extend as needed
    }
    scores = []
    for i in range(len(sequence) - 1):
        pair = (sequence[i], sequence[i + 1])
        scores.append(blosum.get(pair, -4))  # Default score for unknown
    return np.mean(scores) if scores else 0

# Predict PTM sites (basic Ser/Thr near Pro/Gln)
def predict_ptm_sites(sequence):
    return sum(1 for i in range(len(sequence) - 1) if sequence[i] in 'ST' and sequence[i + 1] in 'PQ')

# Estimate interaction energy (placeholder)
def estimate_interaction_energy(sequence):
    return np.random.uniform(-10, 0)

# Main feature extraction
def extract_features(dataset_path):
    df = pd.read_csv(dataset_path)
    
    if 'peptide_seq' not in df.columns:
        raise ValueError("CSV must have a 'peptide_seq' column.")

    df['length'] = df['peptide_seq'].apply(calculate_length)
    df['charge'] = df['peptide_seq'].apply(calculate_charge)
    df['flexibility'] = df['peptide_seq'].apply(estimate_flexibility)
    df['secondary_structure'] = df['peptide_seq'].apply(predict_secondary_structure)
    df['solvent_accessibility'] = df['peptide_seq'].apply(estimate_solvent_accessibility)
    df['blosum_score'] = df['peptide_seq'].apply(get_blosum_scores)
    df['ptm_sites'] = df['peptide_seq'].apply(predict_ptm_sites)
    df['interaction_energy'] = df['peptide_seq'].apply(estimate_interaction_energy)
    
    return df

# Usage
if __name__ == "__main__":
    dataset_path = "test_dataset.csv"  # Make sure this file has 'peptide_seq' column
    result_df = extract_features(dataset_path)
    print(result_df.head())
    result_df.to_csv("enhanced_test_dataset1.csv", index=False)
