import pandas as pd
import hashlib
from pathlib import Path

# Path to the .sums file
sums_file_path = "AML-Cytomorphology.sums"

# Load the .sums file
df = pd.read_csv(sums_file_path, sep=" ", header=None, names=["checksum", "filepath"])

def calculate_md5(file_path, chunk_size=8192):
    """Calculate MD5 checksum of a file in chunks to handle large files."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

# Base directory (where the folders like BAS, MON, etc. are located)
base_dir = Path(".")

# List to hold verification results
results = []

for _, row in df.iterrows():
    file_path = base_dir / row['filepath']
    expected_checksum = row['checksum']
    
    if not file_path.exists():
        results.append((str(file_path), "File not found", False))
        continue
    
    actual_checksum = calculate_md5(file_path)
    match = actual_checksum == expected_checksum
    
    results.append((str(file_path), actual_checksum, match))

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame(results, columns=["filepath", "actual_checksum", "match"])

# Print summary
print(results_df.head())  # Show first 5 rows
print(f"\nTotal files checked: {len(results_df)}")
print(f"Files matching checksum: {results_df['match'].sum()}")
print(f"Files NOT matching checksum: {len(results_df) - results_df['match'].sum()}")

# Optionally, save the results to a CSV file
results_df.to_csv("checksum_verification_results.csv", index=False)


