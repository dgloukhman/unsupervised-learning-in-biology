import torch
import numpy as np
import pandas as pd
import copy
import gzip
import os
import ssl
import urllib.request
from typing import Dict, Tuple, List
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import random

UNIPROT_FASTA_URL = "https://www.uniprot.org/uniprot/{uid}.fasta"
UNIPROT_SEARCH_URL = "https://www.uniprot.org/uniprot/"

SCOP_URL = "https://scop.berkeley.edu/downloads/scopeseq-2.07/astral-scopedom-seqres-gd-sel-gs-bib-40-2.07.fa"
FILENAME = "astral-scopedom-seqres-gd-sel-gs-bib-40-2.07.fa"

# Exclusion Lists for Experiment 3 (from the paper)
ROSSMANN_EXCLUSIONS = ["c.2", "c.3", "c.4", "c.5", "c.27", "c.28", "c.30", "c.31"]
BETA_PROP_EXCLUSIONS = ["b.66", "b.67", "b.68", "b.69", "b.70"]


def randomize_model(model):
    # 1. Move original to CPU temporarily to avoid GPU-copy errors
    device = next(model.parameters()).device
    model.cpu()
    
    # 2. Create the copy safely on CPU
    untrained_model = copy.deepcopy(model)
    
    # 3. Move original back to its device immediately
    model.to(device)
    
    # 4. Randomize weights on the copy
    for name, param in untrained_model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0)
            
    # 5. Return untrained model (caller handles moving it to GPU)
    return untrained_model


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
            results = model(
                batch_tokens, repr_layers=[repr_layer], return_contacts=False
            )

        # Ergebnis extrahieren (Shape: [Batch_Size, Max_Seq_Len_in_Batch, Hidden_Dim])
        token_representations = results["representations"][repr_layer]

        # 5. Ergebnisse speichern und Speicher freigeben
        # WICHTIG: Wir müssen die Tensoren vom GPU-Speicher (CUDA) auf die CPU schieben,
        # sonst läuft der GPU-Speicher voll.
        for j, tokens in enumerate(batch_tokens):
            # Wir entfernen das Padding für jede Sequenz individuell.
            # ESM nutzt Padding Token (oft Index 1). Wir zählen nur echte Token.
            # (Alternativ: Wir nehmen die Länge von batch_strs[j] + 2 für CLS/EOS tokens)
            seq_len = len(batch_strs[j]) + 2  # +2 für <cls> und <eos>

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
        with (
            urllib.request.urlopen(SCOP_URL, context=ctx) as response,
            open(FILENAME, "wb") as out_file,
        ):
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
    with open(FILENAME, "r") as f:
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
    if len(parts) < 2:
        return
    
    #print(parts)

    domain_id = parts[0][1:]
    scop_code = parts[1]
    hierarchy = scop_code.split(".")

    if len(hierarchy) < 4:
        return

    fold = f"{hierarchy[0]}.{hierarchy[1]}"
    superfamily = f"{hierarchy[0]}.{hierarchy[1]}.{hierarchy[2]}"
    family = scop_code

    # FILTERING LOGIC (Phase 1 Rules)
    is_rossmann = any(fold.startswith(ex) for ex in ROSSMANN_EXCLUSIONS)
    is_beta_prop = any(fold.startswith(ex) for ex in BETA_PROP_EXCLUSIONS)

    if not is_rossmann and not is_beta_prop:
        data_list.append(
            {
                "domain_id": domain_id,
                "sequence": sequence,
                "class": hierarchy[0],
                "fold": fold,
                "superfamily": superfamily,
                "family": family,
            }
        )


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
        with gzip.open(response, "rt") as f:
            entry_buffer = []
            for line in f:
                entry_buffer.append(line)
                # End of a record in Stockholm format
                if line.startswith("//"):
                    block = "".join(entry_buffer)
                    # Check if this block matches our target Accession
                    if f"AC   {pfam_accession}" in block:
                        return block
                    entry_buffer = []  # Reset buffer for next entry
    return None


def plot_tsne_exp1(amino_acids_df, representations):
    amino_acids_df = amino_acids_df.copy()
    property_map = {
        "Hydrophobic": ("green", "o"),
        "Aromatic": ("green", "+"),
        "Polar": ("blue", "o"),
        "Unique": ("orange", "o"),
        "Negatively charged": ("red", "x"),
        "Positively charged": ("red", "s"),
    }

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    emb = tsne.fit_transform(representations)
    amino_acids_df["TSNE-1"] = emb[:, 0]
    amino_acids_df["TSNE-2"] = emb[:, 1]
    amino_acids_df
    fig, ax = plt.subplots(figsize=(10, 6))
    for property_name, (color, marker) in property_map.items():
        mask = amino_acids_df["property"].str.capitalize() == property_name
        ax.scatter(
            amino_acids_df[mask]["TSNE-1"],
            amino_acids_df[mask]["TSNE-2"],
            c=color,
            marker=marker,
            label=property_name,
            s=100,
        )

    # Annotate points with amino acid labels
    for i, row in amino_acids_df.iterrows():
        plt.annotate(row["Amino Acid"], (row["TSNE-1"] + 2, row["TSNE-2"]), fontsize=8)

    ax.set_xlabel("TSNE-1")
    ax.set_ylabel("TSNE-2")
    ax.legend()
    plt.show()


def sample_aligned_pairs(
    mapping_list: List[Dict], num_seqs: int, num_samples=50000
) -> List[Tuple[int, int, int, int]]:
    """
    Samples aligned residue pairs from a list of mappings.

    Args:
        mapping_list (List[Dict]): List of mappings with 'query_pos' and 'template_pos'.
        msa_length (int): Length of the MSA.
    """

    aligned_pairs = []
    # --- Sampling Aligned Pairs ---
    # Pick a random column in the MSA, find two sequences that both have a residue there.
    attempts = 0
    while len(aligned_pairs) < num_samples and attempts < num_samples * 5:
        attempts += 1

        # Pick two random sequences
        seq_a_idx = random.randint(0, num_seqs - 1)
        seq_b_idx = random.randint(0, num_seqs - 1)

        if seq_a_idx == seq_b_idx:
            continue

        # Check if both sequences have a residue at this MSA column (no gap)
        map_a = mapping_list[seq_a_idx]
        map_b = mapping_list[seq_b_idx]

        alligned_col = random.choice(list(map_a.keys()))
        alligned_col = int(alligned_col)

        if alligned_col in map_a and alligned_col in map_b:
            # Get the embeddings for these specific residues
            # Note: token_reps includes <cls> at 0, so we add 1 to the mapped index
            aligned_pairs.append((seq_a_idx, seq_b_idx, alligned_col, alligned_col))
    return aligned_pairs


def sample_unaligned_pairs(
    aligned_pairs: List[Tuple[int, int, int, int]],
    mapping_list: List[Dict],
) -> List[Tuple[int, int, int, int]]:
    """
    Samples unaligned residue pairs from a list of mappings and aligned pairs for the background distribution.
    Take into account possible bias because of positional encodings.

    Args:
        mapping_list (List[Dict]): List of mappings with 'query_pos' and 'template_pos'.
        msa_length (int): Length of the MSA.
    """
    unaligned_indices = []
    for seq_a_idx, seq_b_idx, aligned_col, _ in aligned_pairs:
        # 1. Calculate the absolute value of the difference of each residue's within-sequence positions
        map_a = mapping_list[seq_a_idx]
        map_b = mapping_list[seq_b_idx]
        pos_diff = abs(map_a[aligned_col] - map_b[aligned_col])

        attempts_unaligned = 0
        max_attempts = 50000
        while attempts_unaligned < max_attempts:
            attempts_unaligned += 1
            # 2. Select a pair of sequences at random
            seq_c_idx = random.choice(list(range(len(mapping_list))))
            seq_d_idx = random.choice(list(range(len(mapping_list))))

            if seq_c_idx == seq_d_idx:
                continue

            # 3. For that pair of sequences, select a pair of residues at random
            # whose absolute value of positional difference equals the one calculated above
            map_c = mapping_list[seq_c_idx]
            map_d = mapping_list[seq_d_idx]
            reversed_map_d = {v: k for k, v in map_d.items()}

            pos_c = int(random.choice(list(map_c.keys())))
            real_pos_c = map_c[pos_c]

            sign = random.choice([-1, 1])
            real_pos_d = real_pos_c + sign * pos_diff
            if real_pos_d < 0:
                real_pos_d = real_pos_c - sign * pos_diff

            if real_pos_d in reversed_map_d:
                pos_d = reversed_map_d[real_pos_d]

                # 4. Verify that the residues are unaligned (different MSA columns)
                if pos_c != pos_d:
                    unaligned_indices.append((seq_c_idx, seq_d_idx, pos_c, pos_d))
                    break

    return unaligned_indices


def get_cosine_similarity(
    seq_pairs: List[Tuple[int, int, int, int]], token_reps, mapping_list
):
    """
    Computes cosine similarity for given sequence pairs using vectorized operations.

    Args:
        seq_pairs (List[Tuple[int, int, int]]): List of tuples (seq_a_idx, seq_b_idx, aligned_col).

    """
    # Extract embeddings for all pairs at once
    emb_a_list = []
    emb_b_list = []

    for seq_a_idx, seq_b_idx, col_a_idx, col_b_idx in seq_pairs:
        map_a = mapping_list[seq_a_idx]
        map_b = mapping_list[seq_b_idx]

        # +1 because of <cls> token at position 0
        emb_a_list.append(token_reps[seq_a_idx][map_a[col_a_idx] + 1].cpu().numpy())
        emb_b_list.append(token_reps[seq_b_idx][map_b[col_b_idx] + 1].cpu().numpy())

    # Stack into matrices: shape [num_pairs, embedding_dim]

    emb_a = np.array(emb_a_list)
    emb_b = np.array(emb_b_list)

    # Compute all similarities at once
    sims = np.sum(emb_a * emb_b, axis=1) / (
        np.linalg.norm(emb_a, axis=1) * np.linalg.norm(emb_b, axis=1)
    )

    # sims = cosine_similarity(emb_a, emb_b).diagonal()

    return sims.tolist()
