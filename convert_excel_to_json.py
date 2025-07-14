import pandas as pd
import json

file = r"data\Heli Product Details.xlsx"

try:
    df = pd.read_excel(file, engine="openpyxl")
    print(f"✅ Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Rename to consistent internal names
    df = df.rename(columns={
        "Model Name": "Model",
        "Load Capacity (lbs)": "Capacity_lbs",
        "Drive Type": "Type",
        "Power": "Power",
        "Series": "Series",
        "Workplace": "Workplace",
        "Overall Height (in)": "Height_in",
        "Overall Length (in)": "Length_in",
        "Overall Width (in)": "Width_in",  # ✅ corrected spelling here
        "Max Lifting Height (in)": "LiftHeight_in"
    })

    # Convert all columns to object type to allow "N/A"
    df = df.astype("object")
    df.fillna("N/A", inplace=True)

    # Columns to keep in the final JSON
    columns_to_keep = [
        "Model", "Type", "Power", "Capacity_lbs", "Series", "Workplace",
        "Height_in", "Width_in", "Length_in", "LiftHeight_in"
    ]

    # Check for any missing required columns
    missing = [col for col in columns_to_keep if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing column(s): {missing}")

    # Keep only the desired columns
    df = df[columns_to_keep]

    # Convert to JSON
    models_list = df.to_dict(orient="records")

    with open("models.json", "w", encoding="utf-8") as f:
        json.dump(models_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully wrote {len(models_list)} models to models.json")

except Exception as e:
    print("❌ Error:", e)
