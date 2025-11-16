import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from Protein_Encoding import PC_6
from loader import dict_to_array
from explain import shap_explain, integrated_gradients

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------
# CONFIG
# --------------------------
FASTA = "D:/DeepAMP - Copy/test/example.fasta"
LABELS = "D:/DeepAMP - Copy/test/predictions.csv"
MODEL_PATH = "./models/hybrid_transformer_best_weights.h5"


# --------------------------
# LOAD DATA & MODEL
# --------------------------
print("Loading PC6 encoded data...")
pc6_dict = PC_6(FASTA)

labels = pd.read_csv(LABELS, index_col=0)["label"].to_dict()

X, y = dict_to_array(pc6_dict, labels)

print("Loading model...")
model = load_model(MODEL_PATH)

# Pick one sample to explain
sample = X[0:1]

# --------------------------
# SHAP
# --------------------------
print("Running SHAP...")
shap_values = shap_explain(model, sample)

plt.figure(figsize=(16,3))
sns.heatmap(shap_values[0][0], cmap="coolwarm")
plt.title("SHAP - Residue-Level Importance")
plt.xlabel("PC6 Properties")
plt.ylabel("Sequence Position")
plt.show()

# --------------------------
# INTEGRATED GRADIENTS
# --------------------------
print("Running Integrated Gradients...")
ig_values = integrated_gradients(model, sample)

plt.figure(figsize=(16,3))
sns.heatmap(ig_values[0], cmap="viridis")
plt.title("Integrated Gradients Attribution Map")
plt.xlabel("PC6 Properties")
plt.ylabel("Sequence Position")
plt.show()
