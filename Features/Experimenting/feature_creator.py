import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Function to calculate features
def analyze_peptide(sequence):
    sequence = sequence.upper()
    if not sequence.isalpha():
        raise ValueError("Invalid sequence. Only letters A-Z are allowed.")

    analysis = ProteinAnalysis(sequence)

    features = {
        "isoelectric_point": analysis.isoelectric_point(),
        "aromaticity": analysis.aromaticity(),
        "hydrophobicity": analysis.gravy(),
        "stability": analysis.instability_index(),
        "charge": analysis.charge_at_pH(7.0),
        "flexibility": (sequence.count('G') + sequence.count('P')) / len(sequence) if len(sequence) > 0 else 0,
        "solvent_accessibility": np.mean([
            {
                'K': 1.8, 'R': 2.5, 'D': 3.0, 'E': 3.0, 'H': 1.0,
                'A': 0.5, 'C': 0.2, 'F': -2.5, 'I': -3.1, 'L': -2.8,
                'M': -1.9, 'N': 0.2, 'P': 0.0, 'Q': 0.2, 'S': 0.3,
                'T': 0.4, 'V': -2.6, 'W': -3.4, 'Y': -2.3, 'G': 0.0
            }.get(aa, 0) for aa in sequence
        ]) if sequence else 0,
        "blosum_score": np.mean([
            {
                ('A', 'A'): 4, ('A', 'C'): 0, ('A', 'D'): -2, ('A', 'E'): -1,
                ('C', 'C'): 9, ('C', 'A'): 0, ('D', 'D'): 4, ('E', 'E'): 5
            }.get((sequence[i], sequence[i + 1]), -4) for i in range(len(sequence) - 1)
        ]) if len(sequence) > 1 else 0,
        "ptm_sites": sum(1 for i in range(len(sequence) - 1) if sequence[i] in 'ST' and sequence[i + 1] in 'PQ'),
    }

    return features

# Entry point
if __name__ == "__main__":
    user_seq = input("Enter a peptide sequence: ")
    try:
        results = analyze_peptide(user_seq)
        print("\nFeature values:")
        for k, v in results.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"Error: {e}")
