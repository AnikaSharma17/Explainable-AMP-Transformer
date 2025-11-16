#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from Bio import SeqIO
import os

# -------------------------------------------------------------
# 1. Load 6 physicochemical property (PC6) table
# -------------------------------------------------------------
def amino_encode_table_6(path='Data/6_physicochemical_properties/6-pc'):
    """
    Loads PC6 property table from the dataset folder and returns
    a dictionary mapping each amino acid → 6-dim PC6 vector.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"PC6 property file not found: {path}")

    df = pd.read_csv(path, sep=' ', index_col=0)

    # Z-normalize all columns
    columns = ['H1', 'V', 'P1', 'Pl', 'PKa', 'NCI']
    norm = [(df[c] - df[c].mean()) / df[c].std(ddof=1) for c in columns]

    # Combine into a matrix (6 x 20)
    c = np.vstack(norm)

    amino = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    table = {aa: c[:, idx] for idx, aa in enumerate(amino)}

    # Padding token X = zero vector
    table['X'] = np.zeros(6)

    return table


# -------------------------------------------------------------
# 2. Read FASTA
# -------------------------------------------------------------
def read_fasta(fasta_fname):
    """
    Reads a FASTA file and returns dictionary: {id : sequence}
    """
    path = os.path.join("Data/Fasta", fasta_fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"FASTA file not found: {path}")

    r = {}
    for record in SeqIO.parse(path, 'fasta'):
        r[str(record.id)] = str(record.seq).upper()

    return r


# -------------------------------------------------------------
# 3. Pad sequences
# -------------------------------------------------------------
def padding_seq(seq_dict, length=200, pad_value='X'):
    """
    Pads each sequence to fixed length = 200
    Returns: {id : padded_string}
    """

    padded = {}

    for key, seq in seq_dict.items():
        if len(seq) < length:
            seq = seq + pad_value * (length - len(seq))
        else:
            seq = seq[:length]  # trim long sequences
        padded[key] = seq

    return padded


# -------------------------------------------------------------
# 4. PC6 encoding
# -------------------------------------------------------------
def PC_encoding(data_dict):
    """
    Converts padded sequences into PC6 numeric encoding.
    Output: {id : list of 200 vectors (6 dims)}
    """

    table = amino_encode_table_6()
    encoded = {}

    for key, seq in data_dict.items():
        mat = np.array([table.get(aa, table['X']) for aa in seq])
        encoded[key] = mat

    return encoded


# -------------------------------------------------------------
# 5. Full pipeline (FASTA → dict of PC6 matrices)
# -------------------------------------------------------------
def PC_6(fasta_name, length=200):
    """
    Reads FASTA → pads → encodes into PC6.
    Returns: dict {id : (200, 6) numpy array}
    """
    raw = read_fasta(fasta_name)
    padded = padding_seq(raw, length)
    encoded = PC_encoding(padded)
    return encoded


# -------------------------------------------------------------
# 6. For decoding PC6 → FASTA (optional)
# -------------------------------------------------------------
def decode(encoding_value):
    table = amino_encode_table_6()
    for aa, vec in table.items():
        if np.allclose(vec, encoding_value):
            return aa
    return "X"


def re_sequence(embedding_array):
    return ''.join([decode(v) for v in embedding_array])


def data_decoding(data, file_name='decoded'):
    """
    Saves list of PC6 arrays as FASTA
    """
    os.makedirs("Output/Fasta", exist_ok=True)
    out_path = f"Output/Fasta/{file_name}.fasta"

    with open(out_path, 'w') as fasta:
        for i, seq_array in enumerate(data, 1):
            fasta.write(f">sequence_{i}\n")
            fasta.write(re_sequence(seq_array) + "\n")

    print(f"Saved FASTA: {out_path}")
