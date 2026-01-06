import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_tenseal_nn_model(model, train_loader, test_loader, criterion, optimizer,
                           device, num_epochs=50, patience=10, min_delta=1e-5,
                           model_save_path="best_tenseal_nn_model.pth", logger=None):
    """
    Train a PyTorch Neural Network for TenSEAL with early stopping and save the best model.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on ('cpu' or 'cuda').
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        min_delta (float): Minimum improvement to reset patience.
        model_save_path (str): Path to save the best model.
        logger: Logger instance (optional).
    """
    training_losses = []
    evaluation_losses = []
    best_loss = None
    patience_counter = 0
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        training_losses.append(epoch_loss)

        # Evaluation
        model.eval()
        eval_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        eval_loss /= len(test_loader)
        evaluation_losses.append(eval_loss)
        accuracy = 100 * correct / total
        duration = time.time() - start_time

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_save_path)
            if logger:
                logger.info(f"New best model saved at epoch {epoch+1} with Accuracy: {accuracy:.4f}%")

        # Logging
        msg = (f"Epoch {epoch+1}/{num_epochs} | "
               f"Train Loss: {epoch_loss:.4f} | Eval Loss: {eval_loss:.4f} | "
               f"Accuracy: {accuracy:.2f}% | Time: {duration:.2f}s")
        print(msg)
        if logger:
            logger.info(msg)

        # Early Stopping
        if best_loss is None or eval_loss < best_loss - min_delta:
            best_loss = eval_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if logger:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Plot losses
    plt.figure(figsize=(12, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(evaluation_losses, label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

    if logger:
        logger.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    else:
        print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
