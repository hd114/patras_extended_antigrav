import time
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from concrete.ml.sklearn import NeuralNetClassifier, XGBClassifier as ConcreteXGBClassifier, LogisticRegression as ConcreteLogisticRegression
from src.utils.logger import setup_logger

MODEL_MAP = {
    "nn": NeuralNetClassifier,
    "xgb": ConcreteXGBClassifier,
    "lr": ConcreteLogisticRegression
}

def train_concrete_nn_model(X, y, max_epochs=10, n_layers=2, hidden_multiplier=1, logger=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y).astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)

    params = {
        "module__n_layers": n_layers,
        "module__n_hidden_neurons_multiplier": hidden_multiplier,
        "module__activation_function": torch.nn.ReLU,
        "max_epochs": max_epochs,
        "verbose": 0,
        "device": device,
    }

    model = NeuralNetClassifier(**params)

    if logger:
        logger.info(f"Training Concrete-ML NN with params: {params}")

    with tqdm(total=1, desc="Fitting Concrete-ML NN") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    f1_weighted = f1_score(y_test, y_pred, average="weighted") * 100

    if logger:
        logger.info(f"Concrete-ML NN Test Accuracy: {acc:.2f}%")
        logger.info(f"Concrete-ML NN Test F1-Weighted: {f1_weighted:.2f}%")
    else:
        print(f"Concrete-ML NN Test Accuracy: {acc:.2f}%")
        print(f"Concrete-ML NN Test F1-Weighted: {f1_weighted:.2f}%")

    start_compile = time.time()
    model.compile(X_train)
    compile_time = time.time() - start_compile

    if logger:
        logger.info(f"Concrete-ML NN FHE Compile Time: {compile_time:.2f}s")

    return model, label_encoder

def train_concrete_model(X, y, model_type: str, gridsearch: bool = False, logger=None):
    if model_type not in MODEL_MAP:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose from {list(MODEL_MAP.keys())}.")

    ModelClass = MODEL_MAP[model_type]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y).astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)

    if gridsearch and model_type in ["xgb", "lr"]:
        grid_params = get_grid_params(model_type)
        model = GridSearchCV(ModelClass(), grid_params, cv=3, verbose=1, n_jobs=-1)
        model.fit(X_train, y_train)
        best_params = model.best_params_
        if logger:
            logger.info(f"Best params for {model_type}: {best_params}")
        final_model = ModelClass(**best_params)
        final_model.fit(X_train, y_train)
    elif model_type == "nn":
        params = {
            "module__n_layers": 2,
            "module__n_hidden_neurons_multiplier": 1,
            "module__activation_function": torch.nn.ReLU,
            "max_epochs": 10,
            "verbose": 0,
            "device": "cpu"
        }
        final_model = ModelClass(**params)
        if logger:
            logger.info(f"Training Concrete-ML NN with params: {params}")
        with tqdm(total=1, desc=f"Training Concrete-ML {model_type.upper()}") as pbar:
            final_model.fit(X_train, y_train)
            pbar.update(1)
    else:
        final_model = ModelClass()
        final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    f1_weighted = f1_score(y_test, y_pred, average="weighted") * 100
    msg = f"{model_type.upper()} Test Accuracy: {acc:.2f}%"
    f1_msg = f"{model_type.upper()} Test F1-Weighted: {f1_weighted:.2f}%"
    if logger:
        logger.info(msg)
        logger.info(f1_msg)
    else:
        print(msg)
        print(f1_msg)

    start_compile = time.time()
    final_model.compile(X_train)
    compile_time = time.time() - start_compile

    if logger:
        logger.info(f"FHE Compile Time for {model_type.upper()}: {compile_time:.2f}s")

    return final_model, label_encoder

def get_grid_params(model_type: str) -> dict:
    if model_type == "xgb":
        return {
            "n_bits": [8, 10],
            "max_depth": [5, 7],
            "n_estimators": [5, 7]
        }
    elif model_type == "lr":
        return {
            "C": [0.5, 1.0],
            "n_bits": [10, 12],
            "solver": ["lbfgs", "saga"],
            "multi_class": ["auto"]
        }
    else:
        return {}