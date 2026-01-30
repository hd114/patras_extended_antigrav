import sys
import os
# Add project root to path if running from src/
if os.path.basename(os.getcwd()) == "src":
    sys.path.append("..")

import argparse
import torch
from src.utils.logger import setup_logger
from data.data_loader import load_edgeiot_dataset
from src.training.train_tenseal_nn import train_tenseal_nn_model
from src.evaluation.inference_tenseal import run_tenseal_inference
from src.models.mlp import MLPNet
from evaluation.concrete_evaluate import evaluate_concrete_model


# Funktion zum Ausführen des Trainings
def run_tenseal_nn_training():
    logger = setup_logger("training_log")
    npz_file_path = "data/processed/edgeiiot_dataset_all.npz"
    train_loader, val_loader, test_loader, raw_data = load_edgeiot_dataset(npz_file_path, return_raw_data=True)

    # Zugriff auf die Rohdaten und Größe
    train_size = raw_data["X_train"].shape[0]
    val_size = raw_data["X_val"].shape[0]
    test_size = raw_data["X_test"].shape[0]
    total_size = train_size + val_size + test_size

    logger.info(f"Train size: {train_size}")
    logger.info(f"Validation size: {val_size}")
    logger.info(f"Test size: {test_size}")
    logger.info(f"Total dataset size: {total_size}")

    dataset = "EdgeIIoT"
    input_size = raw_data["X_train"].shape[1]
    num_classes = len(raw_data["label_encoder"].classes_)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPNet(input_size=input_size, num_classes=num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_tenseal_nn_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=70,
        patience=15,
        model_save_path="models/CICIoT_best_tenseal_nn.pth",
        logger=logger
    )


def run_concrete_training(model_type, gridsearch):
    logger = setup_logger(f"concrete_{model_type}_log")

    # Daten laden
    npz_file_path = "data/processed/edgeiiot_dataset_all.npz"
    _, _, _, raw = load_edgeiot_dataset(npz_file_path, return_raw_data=True)

    # Testdaten
    X_test, y_test = raw["X_test"], raw["y_test"]

    # Modellpfad für Logging-Zwecke
    model_path = "path/to/saved_model"

    # Evaluierung ausführen
    evaluate_concrete_model(quantized_module, X_test, y_test, model_path=model_path, simulate=True)


def run_evaluation():
    logger = setup_logger("evaluation_log")
    print("Running evaluation...")


def run_fhe_inference():
    logger = setup_logger("fhe_inference_log")
    npz_file_path = "data/processed/edgeiiot_dataset_all.npz"
    _, _, test_loader, raw_data = load_edgeiot_dataset(npz_file_path, return_raw_data=True)

    model_path = "models/plain/CICIoT_best_tenseal_nn.pth"
    context_path = "models/fhe/tenseal_context.con"
    X_test = raw_data["X_test"]
    y_test = raw_data["y_test"]

    run_tenseal_inference(
        model_path=model_path,
        context_path=context_path,
        X_test=X_test,
        y_test=y_test,
        relu_variant="relu2",
        n_samples=20,
        output_csv="results/fhe_tenseal_predictions.csv",
        logger=logger
    )


def test_data_loading(npz_file_path):
    logger = setup_logger("data_loading_test")
    try:
        train_loader, val_loader, test_loader, raw_data = load_edgeiot_dataset(npz_file_path, return_raw_data=True)

        # Überprüfen der Rohdaten und Größe
        train_size = raw_data["X_train"].shape[0]
        val_size = raw_data["X_val"].shape[0]
        test_size = raw_data["X_test"].shape[0]
        total_size = train_size + val_size + test_size

        logger.info(f"Train size: {train_size}")
        logger.info(f"Validation size: {val_size}")
        logger.info(f"Test size: {test_size}")
        logger.info(f"Total dataset size: {total_size}")

        input_size = raw_data["X_train"].shape[1]
        num_classes = len(raw_data["label_encoder"].classes_)

        logger.info(f"Input feature size: {input_size}")
        logger.info(f"Number of classes: {num_classes}")

        print("Data loading test successful.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        print("Data loading test failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control your FHE ML project tasks.")
    parser.add_argument("--task", type=str, required=True,
                        choices=[
                            "tenseal_nn_train",
                            "concrete_train",
                            "evaluate",
                            "fhe_inference",
                            "test_data_loading"
                        ],
                        help="Choose task..."
                        )
    parser.add_argument("--model", type=str, choices=["nn", "xgb", "lr"], help="Concrete-ML model type")
    parser.add_argument("--gridsearch", type=str, choices=["yes", "no"], default="no", help="Perform GridSearch")

    args = parser.parse_args()

    npz_file_path = "data/processed/edgeiiot_dataset_all.npz"

    if args.task == "tenseal_nn_train":
        run_tenseal_nn_training()
    elif args.task == "concrete_train":
        if not args.model:
            print("Please specify --model [nn|xgb|lr] for concrete_train task.")
        elif args.model == "nn" and args.gridsearch == "yes":
            print("Warning: GridSearch is not applicable for NeuralNet. Proceeding without GridSearch.")
            run_concrete_training(args.model, "no")
        else:
            run_concrete_training(args.model, args.gridsearch)
    elif args.task == "evaluate":
        run_evaluation()
    elif args.task == "fhe_inference":
        run_fhe_inference()
    elif args.task == "test_data_loading":
        test_data_loading(npz_file_path)


'''
--task options:      tenseal_nn_train | concrete_train | evaluate | fhe_inference
--model options:     nn | xgb | lr      (nur bei concrete_train)
--gridsearch:        yes | no           (optional, default=no)


python main.py --task tenseal_nn_train
python main.py --task concrete_train --model xgb --gridsearch yes
python main.py --task concrete_train --model lr --gridsearch no
python main.py --task concrete_train --model nn
python main.py --task evaluate
python main.py --task fhe_inference --library tenseal
python main.py --task fhe_inference --library concrete --model xgb
'''
