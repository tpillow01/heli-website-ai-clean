import pandas as pd
import json

file = r"Data\Heli Product Details.xlsx"

try:
    df = pd.read_excel(file, engine="openpyxl")

    print(f"✅ Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
    print("📌 Raw columns:", list(df.columns))

    # Clean up column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Rename to standard keys
    df = df.rename(columns={
        "Model Name": "Model",
        "Drive Type": "Type",
        "Power": "Power",
        "Load Capacity": "Capacity",
        "Series": "Series",
        "Workplace": "Workplace",
        "Overall Height (mm)": "Height_mm",
        "Overal Width (mm)": "Width_mm",
        "Overall Length (mm)": "Length_mm",
        "Max Lifting Height (mm)": "LiftHeight_mm"
    })

    # Select columns to keep
    columns_to_keep = [
        "Model", "Type", "Power", "Capacity", "Series", "Workplace",
        "Height_mm", "Width_mm", "Length_mm", "LiftHeight_mm"
    ]

    missing = [col for col in columns_to_keep if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing column(s): {missing}")

    df = df[columns_to_keep]
    df = df.dropna(subset=["Model", "Type", "Power", "Capacity"])

    models_list = df.to_dict(orient="records")

    with open("models.json", "w", encoding="utf-8") as f:
        json.dump(models_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully wrote {len(models_list)} models to models.json")

except Exception as e:
    print("❌ Error:", e)
