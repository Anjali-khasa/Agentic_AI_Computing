import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

print("\n" + "=" * 85)
print("Part B: Applying Machine Learning Algorithms to Sound Data")
print("=" * 85)

# ------------------------------------------------------------------
# Loading the dataset
# ------------------------------------------------------------------
print("\nStep 1: Loading the dataset...")

current_folder = Path(__file__).resolve().parent
possible_files = [
    current_folder / "sound_data_cleaned.csv",
    current_folder / "sound_data.csv",
    current_folder.parent / "sound_data_cleaned.csv",
    current_folder.parent / "sound_data.csv",
]

data_file = None
for file_path in possible_files:
    if file_path.exists():
        data_file = file_path
        break

if data_file is None:
    print("ERROR: Could not find the CSV file.")
    print("Please keep the CSV either in the same folder as this Python file")
    print("or one folder above it.")
    raise SystemExit

df = pd.read_csv(data_file)
print(f"Dataset loaded successfully from: {data_file}")
print(f"Total rows: {len(df)}")
print(f"Columns available: {list(df.columns)}")

# ------------------------------------------------------------------
# Data Cleaning and Preparation
# ------------------------------------------------------------------
print("\n" + "-" * 85)
print("Step 2: Cleaning and preparing the data")
print("-" * 85)

df.columns = df.columns.str.strip()

text_columns = ["location", "type", "zone_type", "time_block"]
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["decibel_level"] = pd.to_numeric(df["decibel_level"], errors="coerce")

print("Cleaning completed.")
print("\nMissing values by column:")
print(df.isnull().sum())

# Remove rows missing important columns
required_columns = ["timestamp", "location", "type", "time_block", "zone_type"]
df = df.dropna(subset=required_columns).copy()

# ------------------------------------------------------------------
# Creating features
# ------------------------------------------------------------------
print("\n" + "-" * 85)
print("Step 3: Creating meaningful features")
print("-" * 85)

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_name()
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

print("Features created successfully:")
print("- hour")
print("- day_of_week")
print("- is_weekend")
print("- location")
print("- type")
print("- time_block")

# ------------------------------------------------------------------
#  features and target
# ------------------------------------------------------------------
print("\n" + "-" * 85)
print("Step 4: Defining input features and target")
print("-" * 85)

feature_columns = [
    "location",
    "type",
    "time_block",
    "hour",
    "day_of_week",
    "is_weekend"
]

target_column = "zone_type"

X = df[feature_columns].copy()
y = df[target_column].copy()

print(f"Target variable: {target_column}")
print(f"Feature columns: {feature_columns}")

print("\nClass distribution:")
print(y.value_counts())

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ------------------------------------------------------------------
# Splitting the dataset
# ------------------------------------------------------------------
print("\n" + "-" * 85)
print("Step 5: Splitting data into training and testing sets")
print("-" * 85)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
print("\n" + "-" * 85)
print("Step 6: Building the preprocessing pipeline")
print("-" * 85)

categorical_features = ["location", "type", "time_block", "day_of_week"]
numeric_features = ["hour", "is_weekend"]

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("categorical", categorical_transformer, categorical_features),
    ("numeric", numeric_transformer, numeric_features)
])

# ------------------------------------------------------------------
# Trainning models
# ------------------------------------------------------------------
print("\n" + "-" * 85)
print("Step 7: Training machine learning models")
print("-" * 85)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=8),
    "Support Vector Machine": SVC(kernel="rbf", random_state=42)
}

results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "F1 Score": round(f1, 4)
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

# ------------------------------------------------------------------
# Model Comparision
# ------------------------------------------------------------------
print("\n" + "-" * 85)
print("Step 8: Comparing model performance")
print("-" * 85)

results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]
print("\nBest performing model:")
print(f"{best_model['Model']} gave the highest F1 Score of {best_model['F1 Score']:.4f}.")

# ------------------------------------------------------------------
# explanation
# ------------------------------------------------------------------
print("\n" + "-" * 85)
print("Step 9: Interpretation")
print("-" * 85)

print("The models were trained to predict the zone type of each sound observation.")
print("The prediction was based on location, type of area, time block, hour,")
print("day of week, and weekend indicator.")

print("\nImportant note:")
print("Decibel level was not used as an input feature because it is too closely")
print("related to the target variable and could create data leakage.")

print("\nAnalysis complete.")

import matplotlib.pyplot as plt

# results to DataFrame 
results_df = pd.DataFrame(results)

# -----------------------------
# Graph: Model Comparison (Accuracy, Precision, F1)
# -----------------------------
plt.figure(figsize=(10, 6))

x = results_df["Model"]
accuracy = results_df["Accuracy"]
precision = results_df["Precision"]
f1 = results_df["F1 Score"]

plt.plot(x, accuracy, marker='o', label="Accuracy")
plt.plot(x, precision, marker='o', label="Precision")
plt.plot(x, f1, marker='o', label="F1 Score")

for i in range(len(x)):
    plt.text(x[i], accuracy[i] + 0.002, f"{accuracy[i]:.3f}", ha='center')
    plt.text(x[i], precision[i] + 0.006, f"{precision[i]:.3f}", ha='center')
    plt.text(x[i], f1[i] - 0.004, f"{f1[i]:.3f}", ha='center')

plt.title("Model Performance Comparison (Accuracy, Precision, F1 Score)", fontsize=12)
plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Score")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()