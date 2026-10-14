
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = DATA / "Images"
CSV = DATA / "FETAL_PLANES_DB_data.csv"
REPORT = ROOT / "reports" / "split_validation.json"


def main():
    df = pd.read_csv(CSV, sep=";")
    df.columns = df.columns.str.strip()

    # Check the two groups without assuming their meanings yet.
    groups = sorted(df["Train"].dropna().unique().tolist())

    if len(groups) != 2:
        raise ValueError(f"Expected two split indicators; found {groups}")

    group_a = df[df["Train"] == groups[0]]
    group_b = df[df["Train"] == groups[1]]

    patients_a = set(group_a["Patient_num"].dropna())
    patients_b = set(group_b["Patient_num"].dropna())

    overlap = patients_a.intersection(patients_b)

    # Check duplicate image identifiers and missing labels.
    duplicate_images = int(df["Image_name"].duplicated().sum())

    missing_labels = int(
        df["Plane"].isna().sum()
        + df["Patient_num"].isna().sum()
        + df["Train"].isna().sum()
    )

    # Count files that exist but cannot be opened as images.
    from PIL import Image

    unreadable_images = []

    for name in df["Image_name"].dropna().astype(str):
        filename = name.strip()
        if not filename.lower().endswith(".png"):
            filename += ".png"

        path = IMAGES / filename

        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            unreadable_images.append(filename)

    result = {
        "split_indicators": [str(x) for x in groups],
        "split_row_counts": {
            str(value): int(count)
            for value, count in df["Train"].value_counts().items()
        },
        "unique_patients_per_group": {
            str(groups[0]): len(patients_a),
            str(groups[1]): len(patients_b),
        },
        "overlapping_patient_count": len(overlap),
        "overlapping_patient_ids": sorted(
            [str(patient) for patient in overlap]
        ),
        "duplicate_image_identifiers": duplicate_images,
        "missing_required_labels": missing_labels,
        "unreadable_image_count": len(unreadable_images),
        "unreadable_image_examples": unreadable_images[:20],
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("SPLIT VALIDATION COMPLETED")
    print(json.dumps(result, indent=2))
    print(f"\nReport saved to: {REPORT}")


if __name__ == "__main__":
    main()
