import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

def evaluate_tenseal_nn_model(model, data_loader, device, label_encoder, logger=None, plot_cm=False):
    """
    Evaluate a trained TenSEAL NN model on unencrypted data and print common metrics.

    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader for evaluation.
        device: CPU or CUDA device.
        label_encoder: Fitted LabelEncoder for decoding class labels.
        logger: Logger instance (optional).
        plot_cm: Whether to plot the confusion matrix.
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.vstack(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # ROC-AUC
    try:
        roc_auc_macro = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro"
        )
    except ValueError:
        roc_auc_macro = float('nan')

    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Logging or printing
    msg = (
        f"=== Unencrypted Inference Metrics ===\n"
        f"Accuracy:              {acc:.3f}\n"
        f"Precision (macro):     {prec_macro:.3f}\n"
        f"Recall    (macro):     {rec_macro:.3f}\n"
        f"F1-Score  (macro):     {f1_macro:.3f}\n"
        f"F1-Score  (weighted):  {f1_weighted:.3f}\n"
        f"MCC:                   {mcc:.3f}\n"
        f"ROC-AUC (macro, OVR):  {roc_auc_macro:.3f}\n"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)

    print("Classification Report:")
    print(report)

    if plot_cm:
        plot_confusion_matrix(cm, label_encoder.classes_)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot the confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
