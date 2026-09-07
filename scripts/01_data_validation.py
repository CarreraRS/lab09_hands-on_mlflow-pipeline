import mlflow
from sklearn.datasets import load_breast_cancer


def validate_data():
    """Loads the breast cancer dataset, performs validation checks including class balance,

    and logs the results to MLflow.
    """
    # 1. เปลี่ยนชื่อ Experiment ให้สื่อถึง Breast Cancer
    mlflow.set_experiment("Breast Cancer - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # เปลี่ยนไปโหลดข้อมูล Breast Cancer
        cancer_data = load_breast_cancer(as_frame=True)
        df = cancer_data.frame
        print("Data loaded successfully.")

        # 2. ตรวจสอบข้อมูลเบื้องต้น
        num_rows, num_cols = df.shape
        num_classes = df["target"].nunique()
        missing_values = df["target"].isnull().sum()

        # คำนวณสัดส่วนของคลาสน้อยที่สุด (Class Balance Ratio)
        class_proportions = df["target"].value_counts(normalize=True)
        min_class_proportion = class_proportions.min()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")
        print(f"Min class proportion: {min_class_proportion:.4f}")

        # 3. Log ค่า metrics และ parameters ไปยัง MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_metric(
            "class_balance", min_class_proportion
        )  # Log ค่า class_balance เป็น metric

        mlflow.log_param("num_classes", num_classes)

        # 4. เงื่อนไขการตรวจ Validation
        # - ห้ามมี missing_values
        # - จำนวนคลาสต้องเท่ากับ 2 (สำหรับ Breast Cancer)
        # - คลาสน้อยสุดต้องไม่ต่ำกว่า 20% (0.20)
        validation_status = "Success"
        if (
            missing_values > 0
            or num_classes != 2
            or min_class_proportion < 0.20
        ):
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")

        # 5. คืน Exit Code ที่ไม่ใช่ 0 เพื่อให้ CI (GitHub Actions) ตรวจจับความล้มเหลวได้
        if validation_status == "Failed":
            raise SystemExit(
                "Data validation failed — หยุด pipeline ไม่ให้ไปขั้นถัดไป"
            )

        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()