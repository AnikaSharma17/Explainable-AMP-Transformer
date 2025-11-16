# DeepAMP Technical Deep Dive: Components & Architecture

## Table of Contents
1. [Transformer Architecture](#transformer-architecture)
2. [CNN Feature Extractor](#cnn-feature-extractor)
3. [LSTM Decoder](#lstm-decoder)
4. [Loss Functions & Optimization](#loss-functions--optimization)
5. [Regularization Techniques](#regularization-techniques)
6. [Attention Mechanism Details](#attention-mechanism-details)
7. [Complete Model Graph](#complete-model-graph)

---

## Transformer Architecture

### Type: Vanilla Transformer Block (Encoder-style)

Your model uses **standard Transformer blocks** from the original "Attention is All You Need" paper (Vaswani et al., 2017), but **without the decoder**. This is an **encoder-only** or **bidirectional** variant.

### Transformer Block Components

```python
def transformer_block(x, num_heads=4, ff_dim=128):
    # Multi-head self attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    attn_output = Add()([x, attn_output])          # Residual connection
    out1 = LayerNormalization()(attn_output)

    # Feed Forward network
    ffn = Dense(ff_dim, activation="relu")(out1)
    ffn = Dense(x.shape[-1])(ffn)
    ffn_output = Add()([out1, ffn])                # Residual connection
    out2 = LayerNormalization()(ffn_output)

    return out2
```

### What Each Component Does

#### 1. **Multi-Head Self-Attention (MHSA)**

**Purpose:** Learn which parts of the sequence are most important to each position.

**Configuration in Your Model:**
```
num_heads = 4
key_dim = 128  (128 channels per head)
total_dim = 4 × 128 = 512 (after concatenation)
```

**How it works:**
1. Input shape: (batch, 200 positions, 128 channels)
2. Project to Query (Q), Key (K), Value (V) tensors
3. For each of 4 attention heads:
   - Compute attention scores: `Attention(Q, K, V) = softmax(QK^T / √d_k)V`
   - Scale by √(d_k) = √32 = 5.66 (prevents gradient explosion)
4. Concatenate all 4 heads → (batch, 200, 128)

**Why 4 heads?**
- Allows model to attend to different representation subspaces
- More lightweight than single 512-head attention
- Sufficient for peptide sequences (vs. 8+ heads for large language models)

**Mathematical Formula:**
```
head_i = Attention(Q_i, K_i, V_i)
         = softmax(Q_i · K_i^T / √d_k) · V_i

MultiHeadOutput = Concat(head_1, ..., head_4)
```

#### 2. **Residual Connection (Add Layer)**

```python
attn_output = Add()([x, attn_output])
```

**Purpose:** 
- Allows gradients to flow directly through the network (solves vanishing gradient)
- Preserves original signal while adding learned transformations
- Formula: `output = input + f(input)`

**Why important:** Without residual connections, deep networks (4+ layers) suffer from training instability.

#### 3. **Layer Normalization**

```python
LayerNormalization()(attn_output)
```

**Purpose:** Normalize activations across feature dimension for faster training and stability

**Formula:**
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
where:
  μ = mean across features
  σ² = variance across features
  γ, β = learnable scale and shift parameters
  ε = small constant (1e-6) for numerical stability
```

**Location in your block:** 
- After attention + residual (called "post-norm")
- Alternative: pre-norm (before attention) — your model uses post-norm

#### 4. **Feed-Forward Network (FFN)**

```python
ffn = Dense(ff_dim, activation="relu")(out1)      # Expand
ffn = Dense(x.shape[-1])(ffn)                     # Project back
ffn_output = Add()([out1, ffn])                   # Residual
```

**Purpose:** Introduce non-linearity; allow model to learn position-wise transformations

**Structure:**
```
Input (128 dims) → Dense(256, ReLU) → Dense(128) → Output
```

**Expansion ratio:** 256 / 128 = 2× (typical in Transformers is 4×, but yours is more compact)

**Why two Dense layers?**
- First layer (expansion): increases capacity, learns complex patterns
- Second layer (projection): reduces back to original dimension for residual connection
- ReLU activation: introduces non-linearity

**Mathematical formula:**
```
FFN(x) = Linear_2(ReLU(Linear_1(x)))
       = W_2 · max(0, W_1 · x + b_1) + b_2
```

### How Transformer Blocks Are Stacked

In your model:
```python
for _ in range(2):
    x = transformer_block(x, num_heads=4, ff_dim=256)
    x = Dropout(0.3)(x)
```

**Flow:**
```
Input: (batch, 200, 128)
  ↓
[Transformer Block 1]
  → MultiHeadAttention + Residual + LayerNorm
  → FFN + Residual + LayerNorm
  → Output: (batch, 200, 128)
  ↓
[Dropout(0.3)]
  Randomly zeros 30% of activations during training
  ↓
[Transformer Block 2]
  → (batch, 200, 128)
  ↓
[Dropout(0.3)]
```

**Why 2 blocks?**
- 1 block = shallow attention (limited context)
- 2 blocks = moderate depth, good for peptides
- 4+ blocks = overkill for 200-position sequences

### Key Advantages of Transformer

| Aspect | Benefit |
|--------|---------|
| **Parallelizable** | All positions computed simultaneously (vs. RNN's sequential nature) |
| **Long-range dependencies** | Can attend to any position (vs. RNN's limited context) |
| **No gradient vanishing** | Residual connections + LayerNorm |
| **Interpretable** | Attention weights show which residues interact |

---

## CNN Feature Extractor

### Architecture

```python
x = Conv1D(64, 7, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(input_)
x = Dropout(0.4)(x)
x = Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = LayerNormalization()(x)
x = Dropout(0.3)(x)
```

### Component Breakdown

#### Conv1D Layer 1
```
Input:   (batch, 200, 6)
Filters: 64
Kernel:  7 (window size)
Padding: 'same' (pad to preserve length)
Output:  (batch, 200, 64)
```

**What it does:**
- Slides a 7-position window across the sequence
- Learns 64 different local patterns (e.g., hydrophobic motifs, charged clusters)
- Kernel size 7 ≈ captures interactions across ~7 amino acids
- ReLU activation: `f(x) = max(0, x)`

**Example patterns learned:**
- Pattern 1: High PC6[0] (hydrophobic) in window
- Pattern 2: Alternating charge (PC6[5]) pattern
- Pattern 64: Complex combination of properties

#### Dropout(0.4)
- Randomly zeros 40% of the 64 channels during training
- Prevents co-adaptation of features
- At test time: All 64 channels used (output scaled by 1/0.6 to maintain expected value)

#### Conv1D Layer 2
```
Input:   (batch, 200, 64)
Filters: 128
Kernel:  5
Output:  (batch, 200, 128)
```

**Purpose:** 
- Hierarchical feature learning
- Combines patterns from Layer 1 into higher-level features
- Smaller kernel (5 vs. 7) because input already abstracts local structure

**Example learned patterns:**
- Repeating motifs across multiple positions
- Global hydrophobicity distribution
- Charge-hydrophobicity correlations

#### LayerNormalization
- Stabilizes training after combining CNN outputs
- Each position normalized independently

#### Dropout(0.3)
- Slightly less aggressive than after Layer 1 (0.3 vs. 0.4)
- Output ready for Transformer

### Why CNN Before Transformer?

| Reason | Explanation |
|--------|-------------|
| **Computational efficiency** | CNN reduces effective sequence length before attention |
| **Local feature learning** | CNN captures local hydrophobic/charged motifs |
| **Regularization** | Forces model to learn useful low-level features |
| **Hybrid architecture** | Combines strengths: CNN (local structure) + Transformer (global patterns) |

---

## LSTM Decoder

### Architecture

```python
x = LSTM(units=100, return_sequences=False, dropout=0.3)(x)
x = Dropout(0.4)(x)
```

### LSTM Cell Details

**Input shape:** (batch, 200, 128) from Transformer  
**Output shape:** (batch, 100) — single vector per sequence

**Parameters per cell:**
```
Input: 128 features
Hidden: 100 units

Total weights:
  Forget gate:   128×100 + 100×100 + 100 = 21,300
  Input gate:    128×100 + 100×100 + 100 = 21,300
  Cell state:    128×100 + 100×100 + 100 = 21,300
  Output gate:   128×100 + 100×100 + 100 = 21,300
  ──────────────────────────────────────────
  Total:         ~85,200 parameters
```

### How LSTM Works

At each time step t:

```
1. Forget gate: f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
   → Decides which cell state to keep/discard

2. Input gate: i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)
   → Decides which new inputs to keep

3. Candidate state: C̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
   → New information to add

4. Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
   → Update long-term memory (⊙ = element-wise multiply)

5. Output gate: o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)
   → Decides what to output

6. Hidden state: h_t = o_t ⊙ tanh(C_t)
   → Short-term output
```

**Key advantage:** `C_t` (cell state) allows information to flow unchanged across many steps → **no vanishing gradient**

### Why LSTM After Transformer?

| Purpose | Why |
|---------|-----|
| **Sequence summarization** | LSTM's hidden state (h_200) captures final sequence context |
| **Temporal dependencies** | LSTM excels at sequence memory (though Transformer already handles this) |
| **Compatibility** | Output layer expects 1D vector (LSTM outputs flat vector; Transformer outputs 3D) |
| **Sequential processing** | LSTM processes timesteps 1→200 in order |

**Note:** The `return_sequences=False` parameter means:
- Process all 200 timesteps
- Only return the final hidden state (h_200)
- Discards intermediate hidden states

### Dropout in LSTM
```python
dropout=0.3  # Applied to recurrent connections
```

- Drops 30% of connections between LSTM cells
- Prevents co-adaptation of memory cells

---

## Loss Functions & Optimization

### Binary Crossentropy Loss

Your model uses **binary crossentropy** for binary classification (AMP vs. non-AMP):

```python
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**Formula:**
```
L = - [y · log(ŷ) + (1 - y) · log(1 - ŷ)]

where:
  y = true label (0 or 1)
  ŷ = predicted probability from sigmoid (0 to 1)
  log = natural logarithm
```

**Interpretation:**
- If true label y=1 and ŷ is low: Large penalty
- If true label y=0 and ŷ is high: Large penalty
- If prediction matches label: Small penalty

**Why sigmoid output?**
```python
output = Dense(1, activation='sigmoid')(x)
```

Sigmoid function: `σ(z) = 1 / (1 + e^(-z))` maps any real number to (0, 1) range = probability

### Adam Optimizer

```python
optimizer=optimizers.Adam(learning_rate=1e-4)
```

**What it does:**
- Adaptive learning rate per parameter
- Maintains moving averages of gradients (momentum) and squared gradients (RMSprop)
- Learning rate 1e-4 = 0.0001 (small, stable learning)

**Parameter update rule:**
```
m_t = β₁ · m_{t-1} + (1 - β₁) · ∇L         (momentum)
v_t = β₂ · v_{t-1} + (1 - β₂) · (∇L)²     (adaptive scaling)

θ_t = θ_{t-1} - α · m_t / (√v_t + ε)

where:
  β₁ = 0.9 (momentum decay)
  β₂ = 0.999 (squared gradient decay)
  α = learning rate (1e-4)
  ε = 1e-8 (numerical stability)
```

**Why Adam?**
- Converges faster than vanilla SGD
- Works well with small learning rates
- Handles sparse and dense gradients

---

## Regularization Techniques

### 1. L2 Weight Regularization (Ridge)

```python
Conv1D(64, 7, kernel_regularizer=keras.regularizers.l2(1e-4))
Conv1D(128, 5, kernel_regularizer=keras.regularizers.l2(1e-4))
```

**Formula:**
```
Total Loss = Crossentropy Loss + λ · Σ(w²)

where:
  λ = 1e-4 (regularization strength)
  w = each weight in the network
```

**Effect:**
- Penalizes large weights
- Forces model to use smaller, more distributed weights
- Prevents overfitting by reducing model complexity

**Example:**
- Without L2: One feature might have weight = 100 (overfitting)
- With L2: Same feature spreads across multiple smaller weights

### 2. Dropout

```python
# After Conv1D Layer 1
Dropout(0.4)

# After Conv1D Layer 2
Dropout(0.3)

# After Transformer blocks
Dropout(0.3)

# After LSTM
Dropout(0.4)
```

**How it works:**
```
Training:
  output = input · (random_binary_mask) / keep_probability
  Keep probability = 1 - dropout_rate

Example (dropout=0.4):
  Keep probability = 0.6
  For each neuron: 60% chance it outputs value
                    40% chance it outputs 0
  Output scaled by 1/0.6 to maintain expected value

Testing:
  output = input (all neurons active, no scaling)
```

**Why multiple dropout rates?**
- 0.4 after CNN: More aggressive (earlier layers more prone to overfitting)
- 0.3 after Transformer + LSTM: Less aggressive (later layers already regularized by earlier dropout)

### 3. Early Stopping

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)
```

**How it works:**
```
Epoch 1:   val_loss = 0.4500 (best so far)
Epoch 2:   val_loss = 0.4200 (best so far)
...
Epoch 50:  val_loss = 0.0200 (best, patience counter = 0)
Epoch 51:  val_loss = 0.0205 (worse, patience counter = 1)
...
Epoch 80:  val_loss = 0.0210 (no improvement, patience counter = 30)
           → STOP TRAINING

Restore weights from Epoch 50 (best)
```

**Effect:**
- Prevents training past the point of diminishing returns
- Avoids overfitting on training data
- Saves computation time

**Your settings:**
- `patience=30` epochs without improvement → stop
- `restore_best_weights=True` → load best model weights

### 4. Batch Normalization (Implicit)

LayerNormalization in Transformer blocks:
```python
LayerNormalization()
```

- Normalizes feature distributions across channels
- Enables higher learning rates
- Reduces internal covariate shift

---

## Attention Mechanism Details

### Query-Key-Value Framework

Multi-Head Self-Attention breaks down as:

```
Input: X (shape: batch × seq_len × d_model)

Project to Q, K, V:
  Q = X · W_Q  (batch × seq_len × d_k)
  K = X · W_K  (batch × seq_len × d_k)
  V = X · W_V  (batch × seq_len × d_v)

For each head h (4 heads total):
  Attention_h(Q, K, V) = softmax(Q_h · K_h^T / √d_k) · V_h

Concatenate all heads:
  Output = Concat(head_1, ..., head_4) · W_O
```

### Scaled Dot-Product Attention Visualization

```
Query vector for position 100:
  "What features should I attend to?"
  
Key vectors for all positions:
  Position 1:   [similarity: 0.05]
  Position 10:  [similarity: 0.80] ← Highest!
  Position 50:  [similarity: 0.10]
  Position 100: [similarity: 0.05]
  ...
  
Softmax([0.05, 0.80, 0.10, ...]) → [0.01, 0.75, 0.20, ...]
  
Weighted average of Values:
  Output_100 = 0.01×V_1 + 0.75×V_10 + 0.20×V_50 + ...
```

**Interpretation:**
- Position 100 learns most from position 10
- Can skip over intermediate positions
- Captures long-range dependencies (e.g., N-terminal and C-terminal interactions)

---

## Complete Model Graph

### Data Flow

```
Input FASTA
    ↓
[PC6 Encoding]
(200, 6) ← 200 amino acids × 6 properties
    ↓
Input Layer
    ↓
[CNN Block]
Conv1D(64, kernel=7)     → (200, 64)
Dropout(0.4)
Conv1D(128, kernel=5)    → (200, 128)
LayerNormalization
Dropout(0.3)
    ↓
[Transformer Block 1]
MultiHeadAttention(4)    → (200, 128)
Residual + LayerNorm
FFN (256 hidden)         → (200, 128)
Residual + LayerNorm
Dropout(0.3)
    ↓
[Transformer Block 2]
MultiHeadAttention(4)    → (200, 128)
Residual + LayerNorm
FFN (256 hidden)         → (200, 128)
Residual + LayerNorm
Dropout(0.3)
    ↓
[LSTM]
LSTM(100 units)          → (100,)  ← Global sequence representation
Dropout(0.4)
    ↓
[Output Dense Layer]
Dense(1, sigmoid)        → (1,)    ← Probability [0, 1]
    ↓
Output: AMP probability (0.0 - 1.0)
Prediction: if ≥ 0.5 → AMP, else → Non-AMP
```

### Total Parameters

```
CNN Layer 1:      (7×6 + 1) × 64 = 2,816
CNN Layer 2:      (5×64 + 1) × 128 = 40,960

Transformer Block (×2):
  MHSA:           64 × 128 × 4 heads = 32,768
  FFN:            (128×256 + 128) + (256×128 + 128) = 65,792
  Per block:      ~98,560 × 2 = 197,120

LSTM:             85,200

Dense Output:     (100 + 1) × 1 = 101

Total:            ~326,000 parameters
```

### Memory Usage (Training)

```
Batch size: ~50 (50% of dataset)
Input:      50 × 200 × 6 = 60,000 floats
Activations: ~2,000,000 floats (intermediate layers)
Gradients:   ~326,000 floats (one per parameter)
Optimizer states: ~652,000 floats (Adam momentum + squared gradient)

Total:      ~3.5 MB per batch
```

---

## Summary Table: All Components

| Component | Type | Purpose | Key Params |
|-----------|------|---------|-----------|
| **PC6 Encoding** | Feature Engineering | Physicochemical representation | 6 properties, normalized |
| **Conv1D Layer 1** | CNN | Local pattern detection | 64 filters, kernel=7 |
| **Conv1D Layer 2** | CNN | Hierarchical features | 128 filters, kernel=5 |
| **Transformer Block** | Attention | Long-range dependencies | 4 heads, 256 FFN dim |
| **Multi-Head Attention** | Attention Mechanism | Position-wise attention | 4 heads, scaled dot-product |
| **Transformer (×2)** | Stack | Stacked attention + FFN | 2 blocks, residual connections |
| **LSTM** | RNN | Sequence memory | 100 units, returns single vector |
| **Dense Output** | Classification | Binary prediction | 1 neuron, sigmoid activation |
| **L2 Regularization** | Regularization | Weight penalty | λ=1e-4 on CNN |
| **Dropout** | Regularization | Unit dropping | 0.3-0.4 rates |
| **LayerNormalization** | Normalization | Feature normalization | Learnable γ, β |
| **Early Stopping** | Training Control | Prevent overfitting | patience=30 epochs |
| **Adam Optimizer** | Optimization | Adaptive learning | learning_rate=1e-4 |
| **Binary Crossentropy** | Loss | Classification loss | Suitable for binary labels |

---

## Why This Specific Combination?

| Design Choice | Rationale |
|---------------|-----------|
| **Hybrid CNN+Transformer+LSTM** | Captures local (CNN) + global (Transformer) + sequential (LSTM) patterns |
| **4 Transformer heads** | Lightweight, efficient for 200-token sequences |
| **2 Transformer blocks** | Moderate depth; deeper would overfit small datasets |
| **100 LSTM units** | Sufficient for sequence summarization without excess parameters |
| **Dropout 0.3-0.4** | High regularization to combat overfitting on small training sets |
| **L2 regularization 1e-4** | Light weight penalty; prevents explosion but doesn't over-constrain |
| **Early stopping patience=30** | Balanced: stops early enough but allows variability in val_loss |
| **Learning rate 1e-4** | Small enough for stability, large enough for reasonable convergence speed |

---

**End of Technical Deep Dive**
