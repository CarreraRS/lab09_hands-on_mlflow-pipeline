import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    """Simulates a production scenario by loading the Breast Cancer model using the @staging alias

    from MLflow Model Registry and predicting labels for malignant and benign samples.
    """
    # 1. กำหนดชื่อ Tracking URI, โมเดล และ Alias
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")

    # 2. โหลดโมเดลด้วย Alias URI
    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(
            f"Please make sure a model version has the alias '@{MODEL_ALIAS}' in the MLflow UI."
        )
        return

    # 3. โหลดชุดข้อมูล Breast Cancer
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    target_names = cancer.target_names  # ['malignant', 'benign']

    # 4. สุ่ม/เลือกข้อมูลมา 2 แถว (Malignant 1 แถว และ Benign 1 แถว)
    # คลาส 0 คือ malignant, คลาส 1 คือ benign
    malignant_sample = df[df["target"] == 0].iloc[[0]]
    benign_sample = df[df["target"] == 1].iloc[[0]]

    # รวมข้อมูลทั้ง 2 แถวเข้าด้วยกัน
    samples = pd.concat([malignant_sample, benign_sample], axis=0)

    X_sample = samples.drop("target", axis=1)
    y_actual = samples["target"].values

    # 5. ทำนายผลลัพธ์ด้วยโมเดลที่โหลดมา ( Pipeline จะทำ Scaling ให้อัตโนมัติ )
    predictions = model.predict(X_sample)

    # 6. แสดงผลลัพธ์การทำนาย
    print("\n" + "=" * 50)
    print("PRODUCING PREDICTIONS FOR STAGING MODEL")
    print("=" * 50)

    for i in range(len(samples)):
        actual_class = target_names[y_actual[i]]
        predicted_class = target_names[int(predictions[i])]

        print(f"\n[Sample {i + 1}]")
        print(f"  Actual Label   : {y_actual[i]} ({actual_class})")
        print(f"  Predicted Label: {predictions[i]} ({predicted_class})")

        if actual_class == predicted_class:
            print("  Status         : ✅ Correct Prediction")
        else:
            print("  Status         : ❌ Incorrect Prediction")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    load_and_predict()