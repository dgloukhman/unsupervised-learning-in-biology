import torch
import numpy as np
import pandas as pd
import copy
import requests
import gzip
import os
import ssl
import urllib.request
from typing import Dict, Tuple, List

UNIPROT_FASTA_URL = "https://www.uniprot.org/uniprot/{uid}.fasta"
UNIPROT_SEARCH_URL = "https://www.uniprot.org/uniprot/"

SCOP_URL = "https://scop.berkeley.edu/downloads/scopeseq-2.07/astral-scopedom-seqres-gd-sel-gs-bib-40-2.07.fa"
FILENAME = "astral-scopedom-seqres-gd-sel-gs-bib-40-2.07.fa"

# Exclusion Lists for Experiment 3 (from the paper)
ROSSMANN_EXCLUSIONS = ['c.2', 'c.3', 'c.4', 'c.5', 'c.27', 'c.28', 'c.30', 'c.31']
BETA_PROP_EXCLUSIONS = ['b.66', 'b.67', 'b.68', 'b.69', 'b.70']



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

def download_no_requests():
    """Downloads using standard urllib with SSL verification disabled."""
    if os.path.exists(FILENAME):
        print("File already exists. Skipping download.")
        return

    print(f"Downloading {FILENAME} using urllib (SSL ignored)...")
    
    # Create an unverified SSL context to bypass the error
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        with urllib.request.urlopen(SCOP_URL, context=ctx) as response, open(FILENAME, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please try the Manual Download option below.")

def parse_and_filter_scop():
    """Parses FASTA headers and filters out excluded folds."""
    data = []
    
    if not os.path.exists(FILENAME):
        print("Error: File not found.")
        return pd.DataFrame()

    print("Parsing and filtering sequences...")
    with open(FILENAME, 'r') as f:
        current_header = None
        current_seq = []
        
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    process_entry(current_header, "".join(current_seq), data)
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_header:
            process_entry(current_header, "".join(current_seq), data)
            
    return pd.DataFrame(data)

def process_entry(header, sequence, data_list):
    # Header format example: >d1dlwa_ a.4.5.1 (A:)
    parts = header.split()
    if len(parts) < 2: return
    
    domain_id = parts[0][1:]
    scop_code = parts[1]
    hierarchy = scop_code.split('.')
    
    if len(hierarchy) < 4: return
    
    fold = f"{hierarchy[0]}.{hierarchy[1]}"
    superfamily = f"{hierarchy[0]}.{hierarchy[1]}.{hierarchy[2]}"
    family = scop_code
    
    # FILTERING LOGIC (Phase 1 Rules)
    is_rossmann = any(fold.startswith(ex) for ex in ROSSMANN_EXCLUSIONS)
    is_beta_prop = any(fold.startswith(ex) for ex in BETA_PROP_EXCLUSIONS)
    
    if not is_rossmann and not is_beta_prop:
        data_list.append({
            "domain_id": domain_id,
            "sequence": sequence,
            "class": hierarchy[0],
            "fold": fold,
            "superfamily": superfamily,
            "family": family
        })

def get_pfam_seed_by_id(pfam_accession):
    """
    Streams the Pfam-A seed database from EBI FTP and extracts a specific family.
    Bypasses broken API endpoints.
    """
    # Official EBI FTP URL for the current release of Pfam Seed Alignments
    url = "http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.seed.gz"
    
    print(f"Streaming Pfam database to find {pfam_accession}...")
    print("This may take 1-2 minutes as it searches the compressed stream.")

    with urllib.request.urlopen(url) as response:
        # Decompress the stream on the fly
        with gzip.open(response, 'rt') as f:
            entry_buffer = []
            for line in f:
                entry_buffer.append(line)
                # End of a record in Stockholm format
                if line.startswith("//"):
                    block = "".join(entry_buffer)
                    # Check if this block matches our target Accession
                    if f"AC   {pfam_accession}" in block:
                        return block
                    entry_buffer = [] # Reset buffer for next entry
    return None