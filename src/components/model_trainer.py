import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def _save_model(model, out_path="artifacts/model.pkl"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    return out_path

def initiate_model_training(train_arr, test_arr, artifacts_dir="artifacts", use_gridsearch=False):
    try:
        if train_arr is None or test_arr is None:
            raise ValueError("train_arr and test_arr must be provided.")
        if not isinstance(train_arr, np.ndarray) or not isinstance(test_arr, np.ndarray):
            raise ValueError("train_arr and test_arr must be numpy arrays.")
        if train_arr.ndim != 2 or test_arr.ndim != 2:
            raise ValueError("train_arr/test_arr must be 2D arrays.")

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
            raise ValueError("Mismatch between features and target rows.")

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        if use_gridsearch:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
            }
            gs = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            best_params = gs.best_params_
        else:
            model = rf
            model.fit(X_train, y_train)
            best_params = None

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        report = classification_report(y_test, preds)

        os.makedirs(artifacts_dir, exist_ok=True)
        model_path = os.path.join(artifacts_dir, "model.pkl")
        _save_model(model, model_path)

        result = {
            "model_path": model_path,
            "accuracy": acc,
            "f1_weighted": f1,
            "classification_report": report,
            "best_params": best_params,
        }

        print(f"[model_trainer] model saved to: {model_path}")
        print(f"[model_trainer] Accuracy: {acc:.4f}, F1 (weighted): {f1:.4f}")
        print(f"[model_trainer] Classification Report:\n{report}")

        return result

    except Exception as e:
        raise CustomException(e, sys)

class ModelTrainer:
    def __init__(self):
        pass

    def initiate_model_training(self, train_arr, test_arr, artifacts_dir="artifacts", use_gridsearch=False):
        return initiate_model_training(train_arr, test_arr, artifacts_dir=artifacts_dir, use_gridsearch=use_gridsearch)