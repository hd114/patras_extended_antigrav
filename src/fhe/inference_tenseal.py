import tenseal as ts
import torch
import pandas as pd
from src.models.mlp import EncMLPNet

def run_tenseal_inference(
    model_path: str,
    context_path: str,
    X_test,
    y_test,
    relu_variant: str = "relu3",
    n_samples: int = None,
    output_csv: str = "fhe_predictions_tenseal.csv",
    logger=None
):
    """
    Perform FHE inference using TenSEAL on a trained MLP model.

    Args:
        model_path (str): Path to the saved PyTorch model (.pth).
        context_path (str): Path to the TenSEAL context file.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True labels.
        relu_variant (str): Which ReLU approximation to use.
        n_samples (int): Number of samples to predict (if None, use all).
        output_csv (str): Path to save predictions.
        logger: Logger instance (optional).
    """
    device = torch.device('cpu')

    # Load PyTorch model
    input_size = X_test.shape[1]
    num_classes = len(set(y_test))
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Wrap with EncMLPNet
    enc_model = EncMLPNet(model, relu_variant=relu_variant)

    # Load TenSEAL context
    with open(context_path, "rb") as f:
        context = ts.context_from(f.read())

    # Select samples
    if n_samples:
        X_test = X_test[:n_samples]
        y_test = y_test[:n_samples]

    predictions = []
    for i, x in enumerate(X_test):
        enc_x = ts.ckks_vector(context, x.tolist())
        enc_output = enc_model(enc_x)
        decrypted_output = enc_output.decrypt()
        predicted_label = decrypted_output.index(max(decrypted_output))
        predictions.append(predicted_label)

        if logger:
            logger.info(f"Sample {i}: True={y_test[i]}, Predicted={predicted_label}")

    # Save to CSV
    df_results = pd.DataFrame(X_test)
    df_results['true_label'] = y_test
    df_results['predicted_label'] = predictions

    df_results.to_csv(output_csv, index=False)

    msg = f"FHE inference completed. Results saved to {output_csv}"
    logger.info(msg) if logger else print(msg)
