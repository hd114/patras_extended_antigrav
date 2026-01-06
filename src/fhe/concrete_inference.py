import numpy as np
import time
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from datetime import datetime
import os

def run_fhe_inference(quantized_module, X_test, y_test, model_path, simulate=True, n_samples=None):
    """
    Perform FHE inference using Concrete-ML's quantized module.

    Args:
        quantized_module: The quantized module from Concrete-ML.
        X_test (array-like): Test features.
        y_test (array-like): True labels.
        model_path (str): Path to the model file.
        simulate (bool): Whether to simulate the FHE execution or perform it.
        n_samples (int, optional): Number of samples to evaluate. If None, evaluate all.

    Returns:
        None
    """
    mode = "simulate" if simulate else "execute"
    print(f"Running inference in FHE mode: {mode}")

    if not simulate:
        print("Generating key for FHE execution...")
        start_keygen = time.time()
        quantized_module.fhe_circuit.keygen()
        keygen_time = time.time() - start_keygen
        print(f"Key generation completed in {keygen_time:.2f} seconds")

    if n_samples is not None:
        X_test = X_test[:n_samples]
        y_test = y_test[:n_samples]

    y_pred = []
    y_prob = []

    start_inference = time.time()
    for x in tqdm(X_test, desc="FHE Inference", unit="sample"):
        logits = quantized_module.forward(np.expand_dims(x, 0), fhe=mode)
        y_pred.append(np.argmax(logits))
        y_prob.append(softmax(logits[0]))

    inference_time = time.time() - start_inference

    y_pred = np.array(y_pred)
    y_prob = np.vstack(y_prob)

    # Log evaluation metrics
    log_fhe_metrics(y_test, y_pred, y_prob, inference_time, model_path)

def log_fhe_metrics(y_true, y_pred, y_prob, enc_time, model_path):
    """
    Calculate and log various FHE model evaluation metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like): Probability matrix from the FHE model.
        enc_time (float): Total time for encrypted inference.
        model_path (str): Path to the model file.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        roc_auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc_macro = float('nan')

    cm = confusion_matrix(y_true, y_pred)
    cm_str = "\n".join(["[" + " ".join(f"{val:3d}" for val in row) + "]" for row in cm])

    log_text = f"""
=== FHE Model Evaluation ===
Timestamp:              {datetime.now()}
Model path:             {model_path}
Samples evaluated:      {len(y_true)}
Accuracy:               {acc:.3f}
Precision (macro):      {prec_macro:.3f}
Recall    (macro):      {rec_macro:.3f}
F1-Score  (macro):      {f1_macro:.3f}
F1-Score  (weighted):   {f1_weighted:.3f}
MCC:                    {mcc:.3f}
ROC-AUC (macro, OVR):   {roc_auc_macro:.3f}
Total encrypted time:   {enc_time:.1f}s

Confusion Matrix:
{cm_str}

Classification Report:
{classification_report(y_true, y_pred, digits=3, zero_division=0)}

{'='*80}
"""

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    log_dir = "./evaluation_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"eval_{model_name}.log")

    with open(log_filename, "a") as f:
        f.write(log_text)

    print(log_text)
    print(f"📄 Log written to: {log_filename}")