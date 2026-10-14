
from pathlib import Path
import json

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "Images"
CSV_PATH = DATA_DIR / "FETAL_PLANES_DB_data.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "dataset_audit.json"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

    # The dataset CSV uses semicolons as delimiters.
    df = pd.read_csv(CSV_PATH, sep=";")
    df.columns = df.columns.str.strip()

    required_columns = [
        "Image_name",
        "Patient_num",
        "Plane",
        "Brain_plane",
        "Operator",
        "US_Machine",
        "Train",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Build expected image paths and verify file existence.
    image_names = df["Image_name"].astype("string").str.strip()

    def resolve_image_path(name):
        if pd.isna(name) or not name:
            return None

        filename = name if name.lower().endswith(".png") else f"{name}.png"
        return IMAGE_DIR / filename

    image_paths = image_names.map(resolve_image_path)

    missing_image_count = sum(
        path is None or not path.is_file()
        for path in image_paths
    )

    # Keep audit output descriptive; do not assume Train=1 or Train=0
    # means a particular split until we verify the dataset convention.
    report = {
        "dataset_csv": str(CSV_PATH.relative_to(PROJECT_ROOT)),
        "image_directory": str(IMAGE_DIR.relative_to(PROJECT_ROOT)),
        "row_count": int(len(df)),
        "column_names": df.columns.tolist(),
        "missing_values_by_column": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
        },
        "duplicate_metadata_rows": int(df.duplicated().sum()),
        "unique_patients": int(df["Patient_num"].nunique()),
        "plane_class_counts": {
            str(key): int(value)
            for key, value in df["Plane"].value_counts(
                dropna=False
            ).items()
        },
        "brain_plane_counts": {
            str(key): int(value)
            for key, value in df["Brain_plane"].value_counts(
                dropna=False
            ).items()
        },
        "train_column_value_counts": {
            str(key): int(value)
            for key, value in df["Train"].value_counts(
                dropna=False
            ).items()
        },
        "operator_counts": {
            str(key): int(value)
            for key, value in df["Operator"].value_counts(
                dropna=False
            ).items()
        },
        "ultrasound_machine_counts": {
            str(key): int(value)
            for key, value in df["US_Machine"].value_counts(
                dropna=False
            ).items()
        },
        "missing_image_files": int(missing_image_count),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("DATASET AUDIT COMPLETED")
    print(f"Rows: {report['row_count']}")
    print(f"Unique patients: {report['unique_patients']}")
    print(f"Plane classes: {report['plane_class_counts']}")
    print(f"Train indicators: {report['train_column_value_counts']}")
    print(f"Missing image files: {report['missing_image_files']}")
    print(f"Audit report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
