import pandas as pd
import json

# Path to your customer Excel file
file = r"data\SIC Codes.xlsx"  # Adjust this if needed

try:
    df = pd.read_excel(file, engine="openpyxl")
    print(f"✅ Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")

    df.columns = df.columns.str.strip()

    # Convert to JSON format
    records = df.to_dict(orient="records")

    with open("accounts.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"✅ Successfully wrote {len(records)} accounts to accounts.json")

except Exception as e:
    print("❌ Error:", e)
