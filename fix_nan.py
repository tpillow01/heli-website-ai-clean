# fix_nan.py
import json
import math

def replace_nan(obj):
    if isinstance(obj, list):
        return [replace_nan(item) for item in obj]
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if isinstance(v, float) and math.isnan(v):
                new[k] = "N/A"
            else:
                new[k] = replace_nan(v)
        return new
    return obj

# Load the existing JSON
with open("accounts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Replace NaNs
fixed = replace_nan(data)

# Write it back
with open("accounts.json", "w", encoding="utf-8") as f:
    json.dump(fixed, f, indent=2)
