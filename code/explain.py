import numpy as np
import shap
import tensorflow as tf

# -------- SHAP EXPLAINER FOR PC6 MODELS ----------

def shap_explain(model, sample, background=None):
    """
    model: trained Keras model
    sample: np.array shape (1,200,6)
    background: np.array shape (N,200,6)
    """

    if background is None:
        background = np.zeros((20, 200, 6))  # 20 dummy background sequences

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(sample)

    return shap_values


# -------- INTEGRATED GRADIENTS ----------

@tf.function
def interpolate(baseline, sample, alpha):
    return baseline + alpha * (sample - baseline)

def integrated_gradients(model, sample, baseline=None, steps=50):
    """
    sample: (1,200,6)
    baseline: default = zeros
    """
    if baseline is None:
        baseline = np.zeros_like(sample)

    alphas = tf.linspace(0.0, 1.0, steps)
    gradients_list = []

    for alpha in alphas:
        x = interpolate(baseline, sample, alpha)
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = model(x)
        grads = tape.gradient(pred, x)
        gradients_list.append(grads.numpy())

    avg_gradients = np.mean(gradients_list, axis=0)
    integrated_grads = (sample - baseline) * avg_gradients

    return integrated_grads
