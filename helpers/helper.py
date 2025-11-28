import torch
import numpy as np
import copy
import requests
from typing import Dict, Tuple, List

UNIPROT_FASTA_URL = "https://www.uniprot.org/uniprot/{uid}.fasta"
UNIPROT_SEARCH_URL = "https://www.uniprot.org/uniprot/"


def randomize_model(model):
    uninit_model = copy.deepcopy(model)
    for name, param in uninit_model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0)
    return uninit_model


def get_hidden_representations(model, alphabet, labels, sequences, batch_size=1):
    """
    Verarbeitet Sequenzen in Batches durch das ESM-Modell.
    
    Args:
        model: Das geladene ESM Modell.
        alphabet: Das zum Modell gehörige Alphabet-Objekt.
        labels: Liste von Strings (Namen der Proteine).
        sequences: Liste von Strings (Aminosäuresequenzen).
        batch_size: Anzahl der Sequenzen pro Durchlauf (Integer).
        
    Returns:
        tuple: (all_token_representations [List of Tensors], all_strs [List])
    """
    # 1. Gerät bestimmen und Batch Converter vorbereiten
    device = next(model.parameters()).device
    batch_converter = alphabet.get_batch_converter()
    
    # Daten zippen
    data = list(zip(labels, sequences))
    
    all_token_representations = []
    all_strs = []
    
    # 2. Layer identifizieren (Output Layer)
    repr_layer = model.num_layers
    
    print(f"Processing {len(data)} sequences in batches of {batch_size}...")

    # 3. Schleife über die Daten in Chunks (Batches)
    for i in range(0, len(data), batch_size):
        # Slice current batch
        batch_data = data[i : i + batch_size]
        
        # Labels, Strings und Tokens für diesen Batch erzeugen
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        
        # 4. Inferenz (Memory Management ist hier wichtig)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        
        # Ergebnis extrahieren (Shape: [Batch_Size, Max_Seq_Len_in_Batch, Hidden_Dim])
        token_representations = results["representations"][repr_layer]
        
        # 5. Ergebnisse speichern und Speicher freigeben
        # WICHTIG: Wir müssen die Tensoren vom GPU-Speicher (CUDA) auf die CPU schieben,
        # sonst läuft der GPU-Speicher voll.
        for j, tokens in enumerate(batch_tokens):
            # Wir entfernen das Padding für jede Sequenz individuell.
            # ESM nutzt Padding Token (oft Index 1). Wir zählen nur echte Token.
            # (Alternativ: Wir nehmen die Länge von batch_strs[j] + 2 für CLS/EOS tokens)
            seq_len = len(batch_strs[j]) + 2 # +2 für <cls> und <eos>
            
            # Slice: Nur die echten Daten, kein Padding
            # .cpu() bewegt die Daten in den RAM
            seq_rep = token_representations[j, :seq_len].cpu()
            
            all_token_representations.append(seq_rep)
            
        all_strs.extend(batch_strs)
        
        # Optional: Cache leeren, falls GPU sehr klein ist
        torch.cuda.empty_cache()

    return all_token_representations, all_strs


def get_protein_embedding(token_representations, batch_strs):
    """
    Führt Mean Pooling auf den Hidden Representations durch, um ein Embedding pro Protein zu erhalten. 
    TODO: Mean pooling: exclude special tokens & padding.
    
    Args:
        token_representations (List[Tensor]): Output von get_hidden_representations (Liste von Tensoren).
        batch_strs (List[str]): Liste der Sequenzen.
        
    Returns:
        numpy.ndarray: Array der Embeddings (Shape: [N, hidden_dim]).
    """
    embeddings = []
    
    # Iteration durch den Batch
    for i, seq_str in enumerate(batch_strs):
        seq_len = len(seq_str)
        
        # FIX: Zugriff auf Liste zuerst mit [i], dann Slicing des Tensors mit [1 : ...]
        # Wir nehmen Index 1 bis seq_len + 1, um Start-Token (<cls>) und End-Token (<eos>) zu ignorieren.
        # Dies entspricht dem "averaging across the hidden representation at each position".
        
        # token_representations[i] ist der Tensor für das i-te Protein.
        seq_tensor = token_representations[i]
        
        # Slicing: [1 : seq_len + 1] extrahiert nur die Aminosäuren.
        # .mean(0) berechnet den Durchschnitt über die Länge (Dimension 0).
        seq_embedding = seq_tensor[1 : seq_len + 1].mean(0)
        
        embeddings.append(seq_embedding.cpu().numpy())
        
    return np.array(embeddings)