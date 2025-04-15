import pandas as pd
from Bio.SeqUtils import IsoelectricPoint as IP
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Load the Excel file
input_file = "input_F13_test1.xlsx"  # Replace with your file name
df = pd.read_excel(input_file)

# Function to calculate peptide features
def calculate_features(sequence):
    analysis = ProteinAnalysis(sequence)
    try:
        # Calculate isoelectric point
        iso_point = IP.IsoelectricPoint(sequence).pi()
    except:
        iso_point = None
    
    # Calculate aromaticity
    aromaticity = analysis.aromaticity()
    
    # Calculate hydrophobicity (gravy)
    hydrophobicity = analysis.gravy()
    
    # Calculate stability
    stability = analysis.instability_index()
    
    return iso_point, aromaticity, hydrophobicity, stability

# Apply the function to each peptide sequence and create new columns
df[['Isoelectric_Point', 'Aromaticity', 'Hydrophobicity', 'Stability']] = df['peptide_seq'].apply(
    lambda seq: pd.Series(calculate_features(seq))
)

# Save the updated DataFrame to a new Excel file
output_file = "input_F13_test.xlsx"  # Desired output file name
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"Features added and saved to {output_file}")