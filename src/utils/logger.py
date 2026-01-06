import logging
import time
import os
from datetime import datetime

def setup_logger(name: str, log_file: str = None, level=logging.INFO, console_output=True):
    """
    Sets up a logger with specified name, file, level, and console output.

    Args:
    - name (str): Name of the logger.
    - log_file (str, optional): File to log messages to. Defaults to a timestamped file if not provided.
    - level (int): Logging level. Defaults to logging.INFO.
    - console_output (bool): Whether to log messages to console. Defaults to True.

    Returns:
    - logger (logging.Logger): Configured logger.
    """
    if log_file is None:
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def log_training_epoch(logger, epoch, epoch_loss, current_val_loss, f1_weighted, f1_macro_epoch, roc_auc_macro_epoch, start_time):
    """
    Logs the training statistics for a given epoch.

    Args:
    - logger (logging.Logger): Logger to use for logging.
    - epoch (int): Current epoch number.
    - epoch_loss (float): Training loss for the epoch.
    - current_val_loss (float): Validation loss for the epoch.
    - f1_weighted (float): Weighted F1 score for the epoch.
    - f1_macro_epoch (float): Macro F1 score for the epoch.
    - roc_auc_macro_epoch (float): ROC AUC macro score for the epoch.
    - start_time (float): Start time for the epoch.
    """
    # Sichere Formatierung für roc_auc_macro_epoch
    roc_auc_output_string = roc_auc_macro_epoch if isinstance(roc_auc_macro_epoch, str) else f"{roc_auc_macro_epoch:.4f}"

    logger.info(
        f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, "
        f"Val Loss: {current_val_loss:.4f}, F1-weighted: {f1_weighted:.4f}, "
        f"F1-macro: {f1_macro_epoch:.4f}, "
        f"ROC-AUC: {roc_auc_output_string}, " # <--- KORRIGIERTE STELLE
        f"Duration: {time.time() - start_time:.2f}s"
    )

def log_training_summary(logger, model_path, best_f1, fig_path, roc_plot_path, log_path):
    """
    Logs the summary of the training, including the best model and plots.

    Args:
    - logger (logging.Logger): Logger to use for logging.
    - model_path (str): Path where the best model was saved.
    - best_f1 (float): Best weighted F1 score achieved.
    - fig_path (str): Path to the loss plot.
    - roc_plot_path (str): Path to the ROC plot.
    - log_path (str): Path to the detailed training log.
    """
    logger.info(f"✅ Best model saved → {model_path} (weighted_F1: {best_f1:.4f})")
    logger.info(f"📷 Loss plot saved to: {fig_path}")
    logger.info(f"📈 ROC plot for best model saved to: {roc_plot_path}")
    logger.info(f"📄 Training log saved to: {log_path}")

def log_system_info(logger):
    """
    Logs system information including hostname and processor details.

    Args:
    - logger (logging.Logger): Logger to log the system information.
    """
    import platform
    import os

    logger.info(f"Hostname: {platform.node()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Number of Cores: {os.cpu_count()}")