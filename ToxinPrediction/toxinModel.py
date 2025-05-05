import numpy as np
from joblib import load

# Function for calculating AAC (Amino Acid Composition)
def calculate_aac(sequence):
    aac = {aa: sequence.count(aa) / len(sequence) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    return np.array([aac[aa] for aa in 'ACDEFGHIKLMNPQRSTVWY'])

# Function for calculating DPC (Dipeptide Composition)
def calculate_dpc(sequence):
    dipeptides = [sequence[i:i+2] for i in range(len(sequence)-1)]
    dpc = {aa1+aa2: dipeptides.count(aa1+aa2) / len(dipeptides) for aa1 in 'ACDEFGHIKLMNPQRSTVWY' for aa2 in 'ACDEFGHIKLMNPQRSTVWY'}
    return np.array([dpc[aa1+aa2] for aa1 in 'ACDEFGHIKLMNPQRSTVWY' for aa2 in 'ACDEFGHIKLMNPQRSTVWY'])

# Load the pre-trained model
def load_model(model_path):
    return load(model_path)

# Predict the toxin based on the sequence
def predict_toxin(sequence, model, threshold=0.38, model_type=2):
    # Calculate AAC and DPC
    aac = calculate_aac(sequence)
    dpc = calculate_dpc(sequence)
    
    # Combine features for prediction
    if model_type == 1:
        features = np.concatenate((aac, dpc), axis=0)
    else:
        features = np.concatenate((aac, dpc), axis=0)  # Modify based on what model 2 expects

    # Reshape features to match model input
    features = features.reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[:, 1]

    # Adjust threshold based on peptide length
    if len(sequence) < 10:  # If peptide length is less than 10
        threshold = 0.2  # Lower the threshold for smaller peptides

    # Output toxin status
    toxin_status = "Toxin" if probability > threshold else "Non-Toxin"
    
    return toxin_status, probability

# Function to save the output

# Main function to run the process
def main():
    # Hardcoded input data
    sequence = 'ACDEFGHIKLMNPQRSTVWY'  # Example protein sequence (replace with actual sequence)
    output_file = 'output.txt'  # Output file to save results
    threshold = 0.270  # Threshold for toxin prediction (adjust as needed)
    model_type = 2  # Model type (1: AAC&DPC, 2: Hybrid)

    # Load the trained model (update the model path as needed)
    model = load_model('toxinpred3.0_model.pkl')

    # Perform toxin prediction
    toxin_status, probability = predict_toxin(sequence, model, threshold, model_type)

    print(f"Toxin Status: {toxin_status}")
    print(f"Probability: {probability}")

if __name__ == "__main__":
    main()
