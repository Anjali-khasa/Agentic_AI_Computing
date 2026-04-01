import pandas as pd
from pathlib import Path

print("=" * 80)
print("DATA CLEANING AND PROCESSING")
print("=" * 80)

# ---------------------------------------------------
# Load raw sound dataset
# ---------------------------------------------------
current_folder = Path(__file__).resolve().parent
input_file = current_folder / "data.csv"

try:
    df = pd.read_csv(input_file)
    print(f"\nDataset loaded successfully from: {input_file}")
except FileNotFoundError:
    print("\nERROR: data.csv not found.")
    raise SystemExit

print(f"\nInitial number of rows: {len(df)}")
print(f"Initial columns: {list(df.columns)}")

# ---------------------------------------------------
# Standardize column names
# ---------------------------------------------------
df.columns = df.columns.str.strip().str.lower()
print("\nColumn names standardized.")

# ---------------------------------------------------
# Clean text columns
# ---------------------------------------------------
text_columns = ["location", "type", "zone_type", "time_block"]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

print("Text columns cleaned.")

# ---------------------------------------------------
# Convert data types and handle missing values
# ---------------------------------------------------
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

if "decibel_level" in df.columns:
    df["decibel_level"] = pd.to_numeric(df["decibel_level"], errors="coerce")

print("Timestamp and decibel_level converted to correct data types.")

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# ---------------------------------------------------
# Removing duplicates
# ---------------------------------------------------
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()

print(f"\nDuplicate rows removed: {duplicate_count}")
print(f"Rows remaining after duplicate removal: {len(df)}")

# ---------------------------------------------------
# Removing rows with missing values
# ---------------------------------------------------
required_columns = ["timestamp", "location", "decibel_level"]

existing_required = [col for col in required_columns if col in df.columns]
before_drop = len(df)
df = df.dropna(subset=existing_required)

print(f"\nRows removed due to missing required values: {before_drop - len(df)}")
print(f"Rows remaining after removing missing values: {len(df)}")

before_filter = len(df)
df = df[(df["decibel_level"] >= 30) & (df["decibel_level"] <= 120)]

print(f"\nRows removed due to unrealistic decibel values: {before_filter - len(df)}")
print(f"Rows remaining after decibel filtering: {len(df)}")

# ---------------------------------------------------
# Creating additional processed features
# ---------------------------------------------------
if "timestamp" in df.columns:
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    # Create time_block if not already available
    if "time_block" not in df.columns:
        def assign_time_block(hour):
            if 9 <= hour <= 11:
                return "Morning"
            elif 13 <= hour <= 15:
                return "Afternoon"
            else:
                return "Other"

        df["time_block"] = df["hour"].apply(assign_time_block)

print("\nAdditional features created:")
print("- date")
print("- hour")
print("- day_of_week")
print("- is_weekend")
print("- time_block")

# ---------------------------------------------------
# Standardize category labels
# ---------------------------------------------------
if "type" in df.columns:
    df["type"] = df["type"].replace({
        "indoor": "Indoor",
        "outdoor": "Outdoor",
        "Indoor ": "Indoor",
        "Outdoor ": "Outdoor"
    })

if "zone_type" in df.columns:
    df["zone_type"] = df["zone_type"].replace({
        "quiet": "Quiet",
        "moderate": "Moderate",
        "loud": "Loud"
    })

print("Category labels standardized.")

print("\nFinal dataset summary:")
print(f"Final number of rows: {len(df)}")
print(f"Number of locations: {df['location'].nunique() if 'location' in df.columns else 'N/A'}")

print("\nPreview of cleaned dataset:")
print(df.head())

print("\nFinal missing values:")
print(df.isnull().sum())

# ---------------------------------------------------
# Save cleaned dataset
# ---------------------------------------------------
output_file = current_folder / "sound_data.csv"
df.to_csv(output_file, index=False)

