import os
import urllib.request
import zipfile
import pandas as pd

DATA_DIR = "./data"
ZIP_PATH = os.path.join(DATA_DIR, "dataset_diabetes.zip")
CSV_PATH = os.path.join(DATA_DIR, "diabetic_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print("Téléchargement du dataset Diabetes 130-US hospitals...")
    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    urllib.request.urlretrieve(url, ZIP_PATH)
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Téléchargement et décompression terminés !")

df = pd.read_csv(CSV_PATH)
print(f"Dimensions brutes : {df.shape[0]} lignes, {df.shape[1]} colonnes\n")

print("Distribution de la variable cible 'readmitted' :")
print(df['readmitted'].value_counts())
print("\nColonnes disponibles :")
print(df.columns.tolist()[:15])