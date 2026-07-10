from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
csv_path = DATA_DIR / "neuro_mutations.csv"
parquet_path = DATA_DIR / "neuro_mutations.parquet"

if not csv_path.exists():
    raise FileNotFoundError(f"Missing CSV: {csv_path}")

df = None
used_encoding = None

for enc in ["utf-8", "utf-8-sig", "latin1"]:
    try:
        df = pd.read_csv(csv_path, encoding=enc)
        used_encoding = enc
        break
    except Exception:
        pass

if df is None:
    raise RuntimeError("Could not read CSV with utf-8, utf-8-sig, or latin1.")

df.columns = [str(c).strip() for c in df.columns]

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("").astype(str)

df.to_parquet(parquet_path, index=False)

print("Parquet migration complete.")
print(f"CSV source: {csv_path}")
print(f"Encoding used: {used_encoding}")
print(f"Parquet output: {parquet_path}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
