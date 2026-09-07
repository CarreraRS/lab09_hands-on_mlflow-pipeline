import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_evaluate_register(preprocessing_run_id, C=1.0):
    """Loads preprocessed breast cancer data, trains a model, evaluates both

    accuracy and roc_auc, and registers the model if it meets both thresholds.
    """
    # 1. กำหนด Threshold และชื่อโมเดลสำหรับ Breast Cancer
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    ACCURACY_THRESHOLD = 0.95
    ROC_AUC_THRESHOLD = 0.98
    MODEL_NAME = "cancer-classifier-prod"

    mlflow.set_experiment("Breast Cancer - Model Training")

    mlflow.set_experiment("Breast Cancer - Model Training")

    with mlflow.start_run(run_name=f"logistic_regression_C_{C}"):
        print(f"Starting training run with C={C}...")
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        # 2. โหลดข้อมูลจาก Artifacts ของ Preprocessing Run
        try:
            local_artifact_path = download_artifacts(
                run_id=preprocessing_run_id, artifact_path="processed_data"
            )
            print(f"Artifacts downloaded to: {local_artifact_path}")

            train_path = os.path.join(local_artifact_path, "train.csv")
            test_path = os.path.join(local_artifact_path, "test.csv")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print("Successfully loaded data from downloaded artifacts.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            print("Please ensure the preprocessing_run_id is correct.")
            sys.exit(1)

        X_train = train_df.drop("target", axis=1)
        y_train = train_df["target"]
        X_test = test_df.drop("target", axis=1)
        y_test = test_df["target"]

        # 3. สร้างและเทรน Scikit-learn Pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(C=C, random_state=42, max_iter=10000),
            ),
        ])
        pipeline.fit(X_train, y_train)

        # 4. ประเมินผลโมเดล (คำนวณทั้ง Accuracy และ ROC-AUC)
        y_pred = pipeline.predict(X_test)
        # ดึงค่า Probability ของ Positive Class (คลาส 1) สำหรับคำนวณ ROC-AUC
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        # 5. Log Parameters, Metrics และ Model Pipeline เข้า MLflow
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="cancer_classifier_pipeline",
            input_example=X_train.head(5),
        )

        # 6. Gate Check: ตรวจสอบทั้ง 2 เงื่อนไข (Accuracy >= 0.95 AND ROC-AUC >= 0.98)
        if acc >= ACCURACY_THRESHOLD and roc_auc >= ROC_AUC_THRESHOLD:
            print(
                f"Model met all thresholds (Accuracy: {acc:.4f} >= {ACCURACY_THRESHOLD}, "
                f"ROC-AUC: {roc_auc:.4f} >= {ROC_AUC_THRESHOLD}). Registering model..."
            )

            # ลงทะเบียนโมเดลเข้า Model Registry ในชื่อ cancer-classifier-prod
            registered_model = mlflow.register_model(
                model_info.model_uri, MODEL_NAME
            )
            print(
                f"Model registered as '{registered_model.name}' version {registered_model.version}"
            )

            # กำหนด Alias เป็น 'staging'
            client = MlflowClient()
            client.set_registered_model_alias(
                name=MODEL_NAME, alias="staging", version=registered_model.version
            )
            print(
                f"Set alias '@staging' -> {MODEL_NAME} version {registered_model.version}"
            )
        else:
            print(
                f"Model did NOT meet the threshold criteria "
                f"(Accuracy: {acc:.4f}/{ACCURACY_THRESHOLD}, ROC-AUC: {roc_auc:.4f}/{ROC_AUC_THRESHOLD}). "
                f"Not registering."
            )

        print("Training run finished.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python train_evaluate_register.py <preprocessing_run_id> [C_value]"
        )
        sys.exit(1)

    run_id = sys.argv[1]
    c_value = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    train_evaluate_register(preprocessing_run_id=run_id, C=c_value)