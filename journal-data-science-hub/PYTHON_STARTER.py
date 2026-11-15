"""Starter latihan Data Science.
Gunakan dataset milik sendiri atau dataset publik yang lisensinya sesuai.
Install: pip install pandas numpy matplotlib scikit-learn
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Simpan dataset CSV sebagai data.csv, lalu jalankan ulang skrip ini."
    )

df = pd.read_csv(DATA_PATH)

print("Ukuran data:", df.shape)
print("\nLima baris pertama:")
print(df.head())

print("\nTipe data:")
print(df.dtypes)

print("\nRingkasan statistik:")
print(df.describe(include="all"))

print("\nJumlah nilai kosong per kolom:")
print(df.isna().sum())

print("\nJumlah baris duplikat:", df.duplicated().sum())

# Pilih satu kolom numerik yang relevan untuk latihan visualisasi.
numeric_columns = df.select_dtypes(include="number").columns
if len(numeric_columns) > 0:
    column = numeric_columns[0]
    df[column].dropna().hist()
    plt.title(f"Distribusi {column}")
    plt.xlabel(column)
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.show()
else:
    print("Tidak ada kolom numerik untuk visualisasi histogram.")
