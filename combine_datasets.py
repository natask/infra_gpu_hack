from datasets import load_dataset, concatenate_datasets
import pandas as pd

# Load each dataset (adjust the split if necessary)
limo = load_dataset("GAIR/LIMO", split="train")
simplescaling = load_dataset("simplescaling/data_ablation_full59K", split="train")
numinamath = load_dataset("AI-MO/NuminaMath-1.5", split="train")

# Rename "problem" to "question" in the NuminaMath dataset
numinamath = numinamath.rename_column("problem", "question")

# Get the union of all column names across the three datasets
all_columns = set(limo.column_names) | set(simplescaling.column_names) | set(numinamath.column_names)

def add_missing_columns(dataset, all_cols):
    # Determine which columns are missing in the dataset
    missing = list(all_cols - set(dataset.column_names))
    # For each missing column, add it with a default value of None
    for col in missing:
        dataset = dataset.add_column(col, [None] * len(dataset))
    return dataset

# Add any missing columns to each dataset
limo = add_missing_columns(limo, all_columns)
simplescaling = add_missing_columns(simplescaling, all_columns)
numinamath = add_missing_columns(numinamath, all_columns)

# Concatenate datasets in the order: LIMO, simplescaling, then NuminaMath
final_dataset = concatenate_datasets([limo, simplescaling, numinamath])

# Convert the Hugging Face dataset to a Pandas DataFrame and then save as a Parquet file
df = final_dataset.to_pandas()
df.to_parquet("final_dataset.parquet", index=False)

print("Saved combined dataset to final_dataset.parquet")
