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

def get_hidden_representations(model, alphabet, labels, sequences):
    """
    Verarbeitet Sequenzen durch das ESM-Modell und gibt die Hidden States (Token-Repräsentationen) zurück.
    
    Args:
        model: Das geladene ESM Modell.
        alphabet: Das zum Modell gehörige Alphabet-Objekt.
        labels: Liste von Strings (Namen der Proteine).
        sequences: Liste von Strings (Aminosäuresequenzen).
        
    Returns:
        tuple: (token_representations [Tensor], batch_strs [List])
    """
    # 1. Batch Converter vorbereiten
    batch_converter = alphabet.get_batch_converter()
    data = list(zip(labels, sequences))
    
    # 2. Sequenzen in Token umwandeln
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    # 3. Auf das richtige Gerät schieben
    device = next(model.parameters()).device
    batch_tokens = batch_tokens.to(device)

    # 4. Letzten Layer identifizieren
    repr_layer = model.num_layers

    # 5. Inferenz
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
    
    token_representations = results["representations"][repr_layer]
    
    return token_representations, batch_strs

def get_protein_embedding(token_representations, batch_strs):
    """
    Führt Mean Pooling auf den Hidden Representations durch, um ein Embedding pro Protein zu erhalten.
    
    Args:
        token_representations (Tensor): Output von get_hidden_representations.
        batch_strs (List): Liste der Sequenzen (benötigt für die Länge ohne Padding).
        
    Returns:
        numpy.ndarray: Array der Embeddings (Shape: [N, hidden_dim]).
    """
    embeddings = []
    
    # Iteration durch den Batch
    for i, seq_str in enumerate(batch_strs):
        seq_len = len(seq_str)
        
        # Slice [1 : len+1] entfernt Start-Token (CLS) und Padding am Ende.
        # .mean(0) mittelt über die Sequenzlänge (Dimension 0 des Slices).
        # [cite_start]Das entspricht: "averaging across the hidden representation at each position" [cite: 154]
        seq_embedding = token_representations[i, 1 : seq_len + 1].mean(0)
        
        embeddings.append(seq_embedding.cpu().numpy())
        
    return np.array(embeddings)

def fetch_uniprot_sequence(gene, organism):
    # API URL
    url = "https://rest.uniprot.org/uniprotkb/search"
    
    # Suchanfrage: Gen-Name + Organismus + Reviewed (nur kuratierte Swiss-Prot Einträge)
    query = f"gene:{gene} AND organism_name:{organism} AND reviewed:true"
    
    params = {
        "query": query,
        "format": "json",
        "size": 1  # Wir nehmen den besten/ersten Treffer
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['results']:
            # Die Sequenz extrahieren
            sequence = data['results'][0]['sequence']['value']
            entry_id = data['results'][0]['primaryAccession']
            print(f"✅ Found {gene} in {organism} (ID: {entry_id})")
            return sequence
        else:
            print(f"⚠️ No result for {gene} in {organism}")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching {gene}_{organism}: {e}")
        return None