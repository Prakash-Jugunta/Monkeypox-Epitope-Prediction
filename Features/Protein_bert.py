from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

# Load the dataset
df = pd.read_excel('../input_F13_train.xlsx')
sequences = df['peptide_seq'].tolist()
print(sequences)

# Load the model and tokenizer
model_name = "Rostlab/prot_bert_bfd"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# Define function to get embeddings
def get_embeddings(sequences, batch_size=32):
    embeddings = []
    
    # Process sequences in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing Batches"):
        batch_sequences = sequences[i:i + batch_size]
        
        # Tokenize sequences
        tokenized_inputs = tokenizer(batch_sequences, 
                                      return_tensors="pt",
                                      padding=True,
                                      truncation=True,
                                      max_length=512)
        
        # Print tokenized inputs for debugging
       
        
        # Get embeddings without computing gradients
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
        
        # Print model outputs for debugging
        
        # Extract CLS token embeddings
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Print CLS embeddings for debugging

        
        embeddings.append(cls_embedding)

    # Concatenate embeddings and return as NumPy array
    return torch.cat(embeddings, dim=0).numpy()

# Get embeddings for all sequences
cls_embeddings = get_embeddings(sequences, batch_size=32)

# Convert embeddings to DataFrame and concatenate with original DataFrame
embedding_df = pd.DataFrame(cls_embeddings)
df = pd.concat([df, embedding_df], axis=1)

# Save the output to a new Excel file
df.to_excel('output_with_embeddings.xlsx', index=False)
