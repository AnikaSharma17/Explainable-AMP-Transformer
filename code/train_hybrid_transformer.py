import os
import sys
import argparse
import pandas as pd

# Ensure `code/` is on path when running this script from the repo root
sys.path.append(os.path.dirname(__file__))

from Protein_Encoding import PC_6
from loader import dict_to_array
from Model.PC_6_model import t_m


def main(fasta, labels_csv, model_output=None):
    # Load PC6 encodings (returns dict: id -> np.array(200,6))
    pc6 = PC_6(fasta)

    # Load labels from CSV
    df = pd.read_csv(labels_csv)
    
    # Map prediction results to binary labels (1 -> 1, 0 -> 0)
    # Expected columns: Peptide, Score, Prediction
    label_dict = {}
    for idx, row in df.iterrows():
        peptide = row['Peptide']
        prediction = int(row['Prediction'])
        label_dict[peptide] = prediction

    # Convert dicts to arrays
    X_train, y_train = dict_to_array(pc6, label_dict)

    # Train and save model; if model_output not provided, model will be saved under ./models
    model = t_m(X_train, y_train, model_name="hybrid_transformer", path=model_output)

    print(f"Training finished. Best weights saved to: {os.path.join(model_output or os.path.join(os.getcwd(),'models'), 'hybrid_transformer_best_weights.h5')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hybrid Transformer+CNN+LSTM model on PC_6 data')
    parser.add_argument('-f', '--fasta', required=True, help='Filename of training fasta (in Data/Fasta/)')
    parser.add_argument('-l', '--labels', required=True, help='CSV with columns: Peptide, Score, Prediction')
    parser.add_argument('-o', '--model_output', required=False, default=None, help='Directory to save model artifacts (optional)')
    args = parser.parse_args()

    main(args.fasta, args.labels, args.model_output)
