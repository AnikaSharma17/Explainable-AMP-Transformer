import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# keep import local to this script's directory
from PC6_encoding import PC6_encoding

try:
    # prefer tf.keras load if available
    from tensorflow.keras.models import load_model
except Exception:
    from keras.models import load_model


def main(input_fasta_name, output_csv_name, model_path=None, threshold=0.5):
    """Run PC6 predictor.

    Parameters
    - input_fasta_name: path to input fasta file
    - output_csv_name: path to write output CSV
    - model_path: optional path to model (.h5). If not provided, looks for ../model/PC6_final_8.h5 relative to this script.
    - threshold: classification threshold for positive class
    """
    # Resolve paths
    input_fasta = Path(input_fasta_name)
    output_csv = Path(output_csv_name)

    if model_path:
        model_file = Path(model_path)
    else:
        # Default model path for this workspace
        model_file = Path(r'D:\DeepAMP - Copy\code\models\hybrid_transformer_best_weights.h5').resolve()

    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # translate fasta file to PC_6 encode np.array format
    dat = PC6_encoding(str(input_fasta))
    data = np.array(list(dat.values()))

    # load model
    model = load_model(str(model_file), compile=False)

    # predict
    score = model.predict(data)
    classifier = score > float(threshold)

    # make dataframe
    df = pd.DataFrame(score, columns=['Score'])
    df.insert(0, 'Peptide', list(dat.keys()))
    df.insert(2, 'Prediction results', classifier.flatten())
    df['Prediction results'] = df['Prediction results'].replace({True: 'Yes', False: 'No'})

    # ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_csv), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PC6 predictor')
    parser.add_argument('-f', '--fasta_name', help='input fasta name', required=True)
    parser.add_argument('-o', '--output_csv', help='output csv name', required=True)
    parser.add_argument('-m', '--model', help='path to model .h5 (optional)', default=None)
    parser.add_argument('-t', '--threshold', help='classification threshold (default 0.5)', type=float, default=0.5)
    args = parser.parse_args()
    main(args.fasta_name, args.output_csv, model_path=args.model, threshold=args.threshold)