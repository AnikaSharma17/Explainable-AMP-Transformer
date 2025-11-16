# Explainable Transformer Model for Antimicrobial Peptide Prediction: The AI4AMP Framework

A deep learning pipeline for predicting antimicrobial peptides (AMPs) using physicochemical (PC6) encoding and a hybrid Transformer+CNN+LSTM neural network architecture.

---

## Project Overview

**Explainable AMP Transformer** is a machine learning system designed to identify antimicrobial peptides from protein sequences. It combines:

- **PC6 Encoding**: A physicochemical property-based representation of amino acids using 6 properties (Hydrophobicity, Volume, Polarity, Polarizability, pKa, Net Charge Index).
- **Hybrid Deep Learning Model**: A stack of CNN (feature extraction), Transformer (attention-based pattern learning), and LSTM (temporal sequence modeling) layers.
- **Robust Training Framework**: Early stopping, dropout regularization, L2 weight regularization, and class-aware loss weighting.

**Use Cases:**
- Antibiotic drug discovery
- Sequence screening in genome annotation
- Vaccine and immunotherapy research

---

## Architecture & Methodology

### 1. Protein Encoding: PC6 (Physicochemical 6-property)

Each amino acid is encoded as a 6-dimensional vector based on standardized physicochemical properties:

| Property | Abbr. | Description |
|----------|-------|-------------|
| Hydrophobicity | H1 | Measures lipophilicity |
| Molar Volume | V | Amino acid size |
| Polarity | P1 | Polar charge affinity |
| Polarizability | Pl | Electronic deformation |
| Partial Charge (pKa) | PKa | Acid-base properties |
| Net Charge Index | NCI | Overall charge distribution |

**Process:**
1. Load amino acid property matrix from `Data/6_physicochemical_properties/6-pc`
2. Z-normalize each property across the 20 standard amino acids
3. Pad sequences to length 200 (unknown amino acids = zero vector)
4. Output shape: **(N, 200, 6)** where N = batch size

**File:** `code/Protein_Encoding.py::PC_6()`

### 2. Model Architecture: Hybrid Transformer-CNN-LSTM

```
Input (200, 6)
    ↓
[CNN Block]
  - Conv1D(64 filters, kernel=7) + ReLU
  - Dropout(0.4)
  - Conv1D(128 filters, kernel=5) + ReLU
  - LayerNormalization
  - Dropout(0.3)
    ↓
[Transformer Stack × 2]
  - Multi-Head Self-Attention (4 heads)
  - Residual Connections
  - Feed-Forward Network (FFN)
  - LayerNormalization
  - Dropout(0.3)
    ↓
[LSTM Decoder]
  - LSTM(100 units, dropout=0.3)
  - Dropout(0.4)
    ↓
[Output Layer]
  - Dense(1, activation=sigmoid)
    ↓
Output: Score ∈ [0, 1]
```

**Why This Architecture?**

| Component | Purpose |
|-----------|---------|
| **CNN** | Extracts local spatiotemporal patterns from protein sequences |
| **Transformer** | Captures long-range dependencies via multi-head self-attention |
| **LSTM** | Models temporal dynamics and sequence memory |
| **Dropout & L2 Regularization** | Prevents overfitting on small datasets |
| **Early Stopping** | Halts training when validation loss plateaus (patience=30 epochs) |

**File:** `code/Model/PC_6_model.py::t_m()`

### 3. Training Strategy

**Loss Function:** Binary Crossentropy (AMP vs. Non-AMP classification)

**Optimizer:** Adam with learning rate = 1e-4

**Regularization:**
- L2 weight regularization (λ = 1e-4) on CNN layers
- Dropout rates: 0.3–0.4 across layers
- Early stopping with patience=30 epochs
- Validation split: 10%

**Hyperparameters:**
- Batch size: 50% of training set size (adaptive)
- Epochs: 200 (or early stopping)
- Threshold: 0.5 (configurable)

**File:** `code/Model/PC_6_model.py::t_m()`

### 4. Data Pipeline

```
Raw FASTA
    ↓ [PC_6 Encoding]
Dict: {seq_id → np.array(200, 6)}
    ↓ [dict_to_array()]
Training Arrays: X(N, 200, 6), y(N,)
    ↓ [t_m()]
Trained Model + Checkpoints
```

**Key File:** `code/loader.py::dict_to_array()`

---

## Project Structure

```
d:\Explainable AMP Transformer/
├── README.md                          # This file
├── code/
│   ├── Protein_Encoding.py            # PC6 amino acid encoding
│   ├── loader.py                      # Dict→array conversion
│   ├── train_hybrid_transformer.py    # Training entry point (NEW)
│   ├── run_xai.py                     # Explainability analysis
│   ├── model_evalution.py             # Evaluation metrics
│   ├── Data/
│   │   ├── 6_physicochemical_properties/
│   │   │   └── 6-pc                   # PC6 property table
│   │   ├── Fasta/
│   │   │   └── train_positive.fasta   # Training sequences
│   │   ├── labels_train.csv           # Training labels
│   │   └── word2vec/
│   ├── Model/
│   │   └── PC_6_model.py              # Model architecture (UPDATED)
│   └── *.ipynb                        # Legacy Jupyter notebooks
├── PC6/
│   ├── PC6_encoding.py                # Alternative encoding
│   ├── PC6_predictor.py               # Prediction script (UPDATED)
│   ├── evaluate_model.py              # Model evaluation (NEW)
│   └── diagnose_predictor.py          # Debugging tool (NEW)
├── model/
│   ├── PC6_final_8.h5                 # Original trained model
│   └── [other models]
├── code/models/
│   └── hybrid_transformer_best_weights.h5  # New trained model (NEW)
└── test/
    ├── example.fasta                  # 100 test sequences
    ├── example_output.csv             # Reference predictions
    ├── predictions.csv                # Model output
    └── new_predictions.csv            # Latest predictions
```

---

## Installation & Setup

### Prerequisites

- **Python 3.8+**
- **Windows PowerShell 5.1+** (or other terminal)

### Step 1: Create Virtual Environment

```powershell
cd "D:\Explainable AMP Transformer"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
pip install --upgrade pip setuptools wheel
pip install tensorflow pandas biopython keras shap scikit-learn
```

**Core packages:**
- `tensorflow` / `keras` — Deep learning
- `pandas` — Data manipulation
- `biopython` — FASTA parsing
- `numpy` — Numerical computing
- `shap` — Model interpretability
- `scikit-learn` — Utilities (optional)

### Step 3: Verify Installation

```powershell
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

---

## Quick Start

### 1. Train a New Model

```powershell
cd code
python train_hybrid_transformer.py `
  -f train_positive.fasta `
  -l Data/labels_train.csv `
  -o .\models
```

**Expected Output:**
```
Epoch 1/200
2/2 ━━━━━━━━━━━━━━━━━━ 5s 3s/step - accuracy: 0.8889 - loss: 0.6234 - val_accuracy: 1.0000 - val_loss: 0.1234
...
Epoch 200: val_loss improved from X to Y, saving model to .\models\hybrid_transformer_best_weights.h5
Training finished. Best weights saved to: ...
```

**Model saved:** `code/models/hybrid_transformer_best_weights.h5`

### 2. Make Predictions

```powershell
cd ..
python .\PC6\PC6_predictor.py `
  -f .\test\example.fasta `
  -o .\test\predictions.csv
```

**Output CSV:**
```csv
Peptide,Score,Prediction results
AMP_001,0.9924,Yes
AMP_002,0.9926,Yes
NEG_001,0.9917,Yes
...
```

### 3. Evaluate Model Performance

```powershell
python .\PC6\evaluate_model.py `
  -f .\test\example.fasta `
  -l .\test\labels_train.csv `
  -m ".\code\models\hybrid_transformer_best_weights.h5"
```

**Output:**
```
Metrics:
  tp: 45
  tn: 48
  fp: 2
  fn: 5
  accuracy: 0.9300
  precision: 0.9574
  recall: 0.9000
  specificity: 0.9600
  f1: 0.9286
```

---

## Training Pipeline

### Detailed Training Workflow

1. **Load Data**
   ```python
   from Protein_Encoding import PC_6
   from loader import dict_to_array
   
   pc6 = PC_6("train_positive.fasta")  # → Dict[id → (200, 6)]
   labels = {...}  # → Dict[id → {0, 1}]
   X_train, y_train = dict_to_array(pc6, labels)  # → (N, 200, 6), (N,)
   ```

2. **Build Model**
   ```python
   from Model.PC_6_model import t_m
   model = t_m(X_train, y_train, model_name="hybrid_transformer", path="./models")
   ```

3. **Training Process**
   - Compiles with Adam optimizer (lr=1e-4) and binary crossentropy loss
   - Trains up to 200 epochs with 10% validation split
   - Early stops if validation loss doesn't improve for 30 epochs
   - Saves best weights to `{path}/{model_name}_best_weights.h5`
   - Logs training history to `{path}/{model_name}_csvLogger.csv`

4. **Outputs**
   - `hybrid_transformer_best_weights.h5` — Trained model weights
   - `hybrid_transformer_csvLogger.csv` — Training loss/accuracy per epoch

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Input shape | (200, 6) | Padded sequence × PC6 properties |
| CNN filters | 64, 128 | Kernel sizes: 7, 5 |
| Transformer heads | 4 | Multi-head attention |
| Transformer FFN dim | 256 | Feed-forward hidden units |
| LSTM units | 100 | Recurrent hidden state |
| Dropout rates | 0.3–0.4 | Layer-wise regularization |
| L2 weight decay | 1e-4 | CNN kernel regularization |
| Learning rate | 1e-4 | Adam optimizer |
| Batch size | 50% of N | Adaptive |
| Epochs | 200 | Early stopping at patience=30 |
| Validation split | 10% | Hold-out validation set |

---

## Prediction & Evaluation

### Prediction Pipeline

```python
from PC6_encoding import PC6_encoding
from tensorflow.keras.models import load_model
import numpy as np

# 1. Encode sequences
enc = PC6_encoding("test.fasta")  # → Dict[id → (200, 6)]
X = np.array(list(enc.values()))  # → (N, 200, 6)

# 2. Load trained model
model = load_model("hybrid_transformer_best_weights.h5", compile=False)

# 3. Predict
scores = model.predict(X)  # → (N, 1)
predictions = (scores > 0.5).astype(int)  # → {0, 1}
```

### Evaluation Metrics

The evaluation script computes:

- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) — False positive rate
- **Recall (Sensitivity)**: TP / (TP + FN) — True positive rate
- **Specificity**: TN / (TN + FP) — True negative rate
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

**Command:**
```powershell
python .\PC6\evaluate_model.py -f .\test\example.fasta -l .\test\labels_train.csv
```

---

## File Descriptions

### Core Training & Encoding

| File | Purpose |
|------|---------|
| `code/Protein_Encoding.py` | PC6 amino acid encoding; reads FASTA; pads sequences to length 200 |
| `code/loader.py` | Converts PC6 dict → numpy arrays for training |
| `code/Model/PC_6_model.py` | Hybrid Transformer-CNN-LSTM architecture; training loop |
| `code/train_hybrid_transformer.py` | Training entry point; loads data, calls t_m(), saves model |

### Prediction & Evaluation

| File | Purpose |
|------|---------|
| `PC6/PC6_predictor.py` | Inference script; takes FASTA → CSV predictions |
| `PC6/PC6_encoding.py` | Alternative encoding (may differ from `code/Protein_Encoding.py`) |
| `PC6/evaluate_model.py` | Computes accuracy, precision, recall, F1 on labeled data |
| `PC6/diagnose_predictor.py` | Debugging tool; inspects encoding shapes and prediction distributions |

### Data

| Path | Contents |
|------|----------|
| `code/Data/6_physicochemical_properties/6-pc` | PC6 property matrix for 20 amino acids |
| `code/Data/Fasta/` | FASTA files for training |
| `code/Data/labels_train.csv` | Labels: Peptide, Score, Prediction (0/1 or Yes/No) |
| `test/example.fasta` | 100 test sequences (50 AMP + 50 non-AMP) |
| `test/example_output.csv` | Reference predictions with explanations |

### Models

| Path | Description |
|------|-------------|
| `model/PC6_final_8.h5` | Original pre-trained model (legacy) |
| `code/models/hybrid_transformer_best_weights.h5` | New Transformer-CNN-LSTM model (trained with updated architecture) |

---

## Key Hyperparameters

### Model Hyperparameters

```python
# Input
input_shape = (200, 6)

# CNN
conv1_filters = 64
conv1_kernel = 7
conv1_dropout = 0.4

conv2_filters = 128
conv2_kernel = 5
conv2_dropout = 0.3

# Transformer (× 2 blocks)
transformer_heads = 4
transformer_ffn_dim = 256
transformer_dropout = 0.3

# LSTM
lstm_units = 100
lstm_dropout = 0.3
lstm_output_dropout = 0.4

# Training
optimizer = Adam(learning_rate=1e-4)
loss = binary_crossentropy
l2_regularizer = 1e-4
early_stop_patience = 30
validation_split = 0.1
epochs = 200
batch_size = ceil(N * 0.5)
```

### Prediction Hyperparameters

```python
threshold = 0.5  # Score above threshold → Predicted as AMP
```

To adjust threshold when running prediction:
```powershell
python .\PC6\PC6_predictor.py -f .\test\example.fasta -o out.csv -t 0.6
```

---

## Troubleshooting

### Issue: "Model file not found"

**Cause:** Model path is incorrect or file doesn't exist.

**Solution:**
```powershell
# Check file exists
Test-Path "D:\DeepAMP - Copy\code\models\hybrid_transformer_best_weights.h5"

# Specify explicit path
python .\PC6\PC6_predictor.py -f .\test\example.fasta -o .\test\predictions.csv `
  -m "D:\DeepAMP - Copy\code\models\hybrid_transformer_best_weights.h5"
```

### Issue: "FASTA file not found"

**Cause:** PC_6 function expects filename, not full path.

**Solution:**
```powershell
# Use filename only (PC_6 prepends "Data/Fasta/")
python train_hybrid_transformer.py -f train_positive.fasta -l Data/labels_train.csv -o models
```

### Issue: All predictions are "Yes" or "No"

**Cause:** Model is poorly trained or dataset is heavily imbalanced.

**Debugging:**
```powershell
python .\PC6\diagnose_predictor.py -f .\test\example.fasta
```

Check output for:
- `Encoded shape`: Should be (N, 200, 6)
- `Preds min/max/mean`: If all >0.9, model may be overfit or unbalanced data
- `Preds range`: Should span [0.0, 1.0] for a properly trained model

**Fix:**
- Retrain with regularization (dropout, L2)
- Use balanced dataset or class weighting
- Adjust threshold based on precision-recall tradeoff

### Issue: Installation errors for `shap` or `tensorflow`

**Solution:** Use specific compatible versions
```powershell
pip install tensorflow==2.13.0 keras==2.13.0 shap==0.41.0
```

Or use CPU-only TensorFlow:
```powershell
pip install tensorflow-cpu pandas biopython
```

---

## Citation & References

This project builds on:
- **Transformer architecture**: Vaswani et al. (2017) "Attention is All You Need"
- **Physicochemical properties**: Ippel et al. (1992); Sandberg et al. (1998)
- **Antimicrobial peptide research**: Historical work in computational proteomics


---
