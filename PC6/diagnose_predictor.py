import numpy as np
from pathlib import Path
import argparse
from PC6_encoding import PC6_encoding

try:
    from tensorflow.keras.models import load_model
except Exception:
    from keras.models import load_model


def diagnose(fasta_path, model_path=None, n_show=5):
    fasta = Path(fasta_path)
    if model_path:
        model_file = Path(model_path)
    else:
        model_file = Path(r'D:\DeepAMP - Copy\code\models\hybrid_transformer_best_weights.h5')

    print('FASTA:', fasta)
    print('Model:', model_file)
    if not fasta.exists():
        print('ERROR: fasta not found')
        return
    if not model_file.exists():
        print('ERROR: model not found')
        return

    enc = PC6_encoding(str(fasta))
    keys = list(enc.keys())
    arr = np.array(list(enc.values()))
    print('Encoded shape:', arr.shape)
    print('Encoded dtype:', arr.dtype)
    print('Encoded min/max:', arr.min(), arr.max())
    print('Encoded mean/std:', arr.mean(), arr.std())

    # show first rows
    print('\nFirst encoded rows:')
    for i in range(min(n_show, arr.shape[0])):
        print(keys[i], arr[i].flatten()[:10])

    model = load_model(str(model_file), compile=False)
    print('\nModel summary:')
    try:
        model.summary()
    except Exception as e:
        print('Could not print model summary:', e)

    preds = model.predict(arr)
    print('\nPredictions shape:', preds.shape)
    print('Preds min/max/mean:', preds.min(), preds.max(), preds.mean())
    print('First predictions:')
    for i in range(min(n_show, preds.shape[0])):
        print(keys[i], preds[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--fasta', required=True)
    parser.add_argument('-m','--model', required=False)
    args = parser.parse_args()
    diagnose(args.fasta, args.model)
