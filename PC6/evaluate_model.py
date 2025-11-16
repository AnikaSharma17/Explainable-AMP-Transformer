import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from PC6_encoding import PC6_encoding
try:
    from tensorflow.keras.models import load_model
except Exception:
    from keras.models import load_model


def read_label_csv(labels_csv):
    df = pd.read_csv(labels_csv)
    # Possible columns: 'Peptide','Prediction' (0/1) or 'Prediction results' ('Yes'/'No')
    if 'Peptide' not in df.columns:
        # try first column as peptide
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: 'Peptide'})
    if 'Prediction' in df.columns:
        labels = {row['Peptide']: int(row['Prediction']) for _, row in df.iterrows()}
    elif 'Prediction results' in df.columns:
        def mapval(v):
            if isinstance(v, str):
                return 1 if v.strip().lower() in ('yes','y','1','true','t') else 0
            return int(v)
        labels = {row['Peptide']: mapval(row['Prediction results']) for _, row in df.iterrows()}
    else:
        # try last column as label
        lastcol = df.columns[-1]
        def mapval(v):
            if isinstance(v, str):
                return 1 if v.strip().lower() in ('yes','y','1','true','t') else 0
            return int(v)
        labels = {row['Peptide']: mapval(row[lastcol]) for _, row in df.iterrows()}
    return labels


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(int).flatten()
    y_pred = np.array(y_pred).astype(int).flatten()
    assert y_true.shape == y_pred.shape
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'specificity': specificity, 'f1': f1
    }


def main(fasta, labels_csv, model_path=None, threshold=0.5):
    fasta = Path(fasta)
    labels_csv = Path(labels_csv)
    if model_path:
        model_file = Path(model_path)
    else:
        model_file = Path(r'D:\DeepAMP - Copy\model\PC6_final_8.h5')

    if not fasta.exists():
        print('FASTA not found:', fasta)
        return
    if not labels_csv.exists():
        print('Labels CSV not found:', labels_csv)
        return
    if not model_file.exists():
        print('Model not found:', model_file)
        return

    print('Loading and encoding sequences...')
    enc = PC6_encoding(str(fasta))
    keys = list(enc.keys())
    X = np.array(list(enc.values()))
    print('Encoded shape:', X.shape)

    print('Loading model:', model_file)
    model = load_model(str(model_file), compile=False)
    print('Predicting...')
    probs = model.predict(X).flatten()
    preds = (probs > float(threshold)).astype(int)

    labels = read_label_csv(labels_csv)
    y_true = []
    y_pred = []
    missing = []
    for i, k in enumerate(keys):
        if k in labels:
            y_true.append(labels[k])
            y_pred.append(int(preds[i]))
        else:
            missing.append(k)

    print(f'Found {len(y_true)} labeled sequences, {len(missing)} missing labels')
    if missing:
        print('Missing examples (first 10):', missing[:10])

    metrics = compute_metrics(y_true, y_pred)
    print('\nMetrics:')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta', required=True)
    parser.add_argument('-l', '--labels', required=True)
    parser.add_argument('-m', '--model', required=False)
    parser.add_argument('-t', '--threshold', required=False, type=float, default=0.5)
    args = parser.parse_args()
    main(args.fasta, args.labels, args.model, args.threshold)
