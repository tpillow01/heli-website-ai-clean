import json
from pathlib import Path
import pandas as pd

# --- paths ---
EXCEL_PATH = Path(r"data\Heli Product Details.xlsx")  # adjust if needed
SHEET_NAME = 0  # use 0 for first sheet, or put the sheet name string here
OUTPUT_JSON = Path("models.json")

def clean_headers(cols):
    fixed = []
    for c in cols:
        name = "" if c is None else str(c)
        name = name.replace("\u00A0", " ")          # normalize NBSP
        name = " ".join(name.split())               # collapse internal spaces
        fixed.append(name.strip())
    return fixed

def main():
    # Read as strings so values like '49-57' stay intact
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl", dtype=str)
    print(f"✅ Loaded Excel with {len(df)} rows and {len(df.columns)} columns")

    # Keep ALL columns; just clean headers/whitespace
    df.columns = clean_headers(df.columns)

    # Trim cell strings, keep non-strings as-is
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop fully empty rows/columns, dedupe columns
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Fill remaining empties with "N/A"
    df = df.fillna("N/A")

    # Show exactly what headers will be written
    print("🧾 Columns going to JSON:")
    for c in df.columns:
        print("  •", c)

    # Write list-of-records JSON with ALL columns
    records = df.to_dict(orient="records")
    OUTPUT_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Successfully wrote {len(records)} models to {OUTPUT_JSON.resolve()}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error:", e)
