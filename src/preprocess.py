import pandas as pd
import numpy as np
import torch

CSV_PATH = "./data/diabetic_data.csv"
df = pd.read_csv(CSV_PATH)
df = df.drop_duplicates(subset=["patient_nbr"], keep="first").copy()

df["target"] = (df["readmitted"] == "<30").astype(int)

feature_names = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

X_df = df[feature_names]
Y_series = df["target"]

print("--- Vérification des valeurs manquantes ---")
print(X_df.isnull().sum())

print(f"\nDimensions de X : {X_df.shape[0]} exemples (N), {X_df.shape[1]} variables (m)")
print("Statistiques descriptives rapides :")
print(X_df.describe().T[["mean", "std", "min", "max"]])

X_raw = X_df.to_numpy(dtype=np.float32)
Y_raw = Y_series.to_numpy(dtype=np.float32)

np.random.seed(42)

N = X_raw.shape[0]
indices = np.random.permutation(N)

split_idx = int(0.8 * N)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train_raw = X_raw[train_indices]
Y_train_raw = Y_raw[train_indices]

X_test_raw = X_raw[test_indices]
Y_test_raw = Y_raw[test_indices]

mean = np.mean(X_train_raw, axis=0)
std = np.std(X_train_raw, axis=0)

std = np.where(std == 0, 1.0, std)

X_train_norm = (X_train_raw - mean) / std
X_test_norm = (X_test_raw - mean) / std

X_train = torch.from_numpy(X_train_norm)
X_test = torch.from_numpy(X_test_norm)

Y_train = torch.from_numpy(Y_train_raw).unsqueeze(1)
Y_test = torch.from_numpy(Y_test_raw).unsqueeze(1)

print("\n--- Dimensions des Tenseurs PyTorch finaux ---")
print(f"X_train : {X_train.shape}  | dtype : {X_train.dtype}")
print(f"Y_train : {Y_train.shape}  | dtype : {Y_train.dtype}")
print(f"X_test  : {X_test.shape}   | dtype : {X_test.dtype}")
print(f"Y_test  : {Y_test.shape}   | dtype : {Y_test.dtype}")