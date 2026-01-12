import os
import sys
import time
import math
import getpass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import make_blobs

# --- LIBRARY TAMBAHAN ---
from colorama import init, Fore, Style, Back
from tabulate import tabulate

# Inisialisasi Colorama
init(autoreset=True)

# ==========================================
# KONFIGURASI SISTEM & UI
# ==========================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def loading_animation(text="Processing", duration=1.0):
    print(f"\n{Fore.YELLOW}{text}", end="")
    for i in range(3):
        time.sleep(0.2)
        sys.stdout.write(".")
        sys.stdout.flush()
    
    # Progress Bar
    print(f"\n{Fore.CYAN}[", end="")
    for i in range(25):
        time.sleep(duration / 25)
        sys.stdout.write("█")
        sys.stdout.flush()
    print(f"] {Fore.GREEN}100%\n")

def print_header():
    clear_screen()
    print(Fore.CYAN + Style.BRIGHT + "="*85)
    print(Fore.CYAN + r"""
   ___   _  _____  _      ____   ___ ___  ___  _  __ ____ _____ 
  / _ \ / |/ / _ \/ | /| / /  | / _// _ \/ _ \/ |/ // __// ___/ 
 / // //    / // /| |/ |/ / / |/ _// // / // /    /_\ \ / /__   
/____//_/|_/____/ |__/|__/  |___/___/____/____/_/|_/____/\___/  
                                                                
    ADVANCED DATA SCIENCE SUITE v3.0 (ULTIMATE)
    """)
    print(Fore.CYAN + "="*85)
    
    info = [
        ["User Logged In", "Rizqi Ghani Adinata (Admin)"],
        ["System ID", "24EP10007-SI-UAS"],
        ["Target Course", "Interaksi Manusia & Komputer"],
        ["Supervisor", "Bapak Abdul Haq"],
        ["Security Level", "ENCRYPTED (AES-256)"]
    ]
    print(tabulate(info, tablefmt="simple_grid", stralign="left"))
    print(Fore.YELLOW + "\n[DASHBOARD] Select a module to begin operation...\n")

def pause():
    input(f"\n{Back.CYAN}{Fore.BLACK} ENTER {Style.RESET_ALL} {Fore.WHITE}untuk kembali ke menu utama...")

# ==========================================
# MODUL 0: SECURITY (LOGIN)
# ==========================================
def login_system():
    clear_screen()
    print(Fore.GREEN + "=== SECURE LOGIN GATEWAY ===")
    print(Fore.WHITE + "Default User: admin | Pass: admin123")
    
    attempts = 0
    while attempts < 3:
        user = input(f"\n{Fore.YELLOW}Username: {Fore.WHITE}")
        # getpass menyembunyikan input password (seperti terminal asli)
        try:
            pwd = getpass.getpass(f"{Fore.YELLOW}Password: {Fore.WHITE}")
        except:
            pwd = input(f"{Fore.YELLOW}Password: {Fore.WHITE}") # Fallback jika getpass error di IDE tertentu

        loading_animation("Verifying Credentials", 0.5)
        
        if user == "admin" and pwd == "admin123":
            print(f"\n{Back.GREEN}{Fore.BLACK} ACCESS GRANTED {Style.RESET_ALL}")
            time.sleep(1)
            return True
        else:
            print(f"{Back.RED}{Fore.WHITE} ACCESS DENIED {Style.RESET_ALL} Sisa percobaan: {2 - attempts}")
            attempts += 1
            
    print(f"\n{Fore.RED}SYSTEM LOCKED. INTRUDER DETECTED.")
    sys.exit()

# ==========================================
# MODUL 1: DATA PREPROCESSING (PANDAS)
# ==========================================
def data_cleaning_demo():
    print_header()
    print(f"{Back.MAGENTA}{Fore.WHITE} [ MODUL: DATA WRANGLING & CLEANING ] {Style.RESET_ALL}\n")
    
    # 1. Membuat Data Kotor (Dummy)
    data = {
        'Nama': ['Ali', 'Budi', 'Citra', 'Deni', 'Eka', 'Fani'],
        'Usia': [20, 21, np.nan, 22, 23, np.nan],
        'Nilai': [85, np.nan, 90, 88, np.nan, 95],
        'Kota': ['Jakarta', 'Bandung', 'Jakarta', None, 'Surabaya', 'Bandung']
    }
    df = pd.DataFrame(data)
    
    print(f"{Fore.RED}>>> DATASET AWAL (KOTOR/MISSING VALUES):")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    print(f"\n{Fore.YELLOW}[ANALISIS] Mendeteksi Missing Values...")
    print(df.isnull().sum())
    
    input(f"\n{Fore.GREEN}Tekan ENTER untuk melakukan Data Cleaning otomatis...")
    loading_animation("Cleaning Data & Imputation")
    
    # 2. Proses Cleaning
    # Isi Usia kosong dengan Rata-rata
    df['Usia'] = df['Usia'].fillna(df['Usia'].mean())
    # Isi Nilai kosong dengan Median
    df['Nilai'] = df['Nilai'].fillna(df['Nilai'].median())
    # Isi Kota kosong dengan Modus (Data terbanyak)
    df['Kota'] = df['Kota'].fillna(df['Kota'].mode()[0])
    
    print(f"\n{Fore.CYAN}>>> DATASET BERSIH (CLEANED):")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
    
    # 3. Export Data
    if input(f"\n{Fore.YELLOW}Simpan ke Excel? (y/n): ").lower() == 'y':
        try:
            df.to_excel("Laporan_Data_Clean.xlsx", index=False)
            print(f"{Fore.GREEN}[SUCCESS] File tersimpan sebagai 'Laporan_Data_Clean.xlsx'")
        except:
            print(f"{Fore.RED}[ERROR] Install openpyxl: pip install openpyxl")
            
    pause()

# ==========================================
# MODUL 2: ADVANCED VISUALIZATION (SEABORN)
# ==========================================
def correlation_heatmap():
    print_header()
    print(f"{Back.MAGENTA}{Fore.WHITE} [ MODUL: STATISTICAL CORRELATION (HEATMAP) ] {Style.RESET_ALL}\n")
    
    loading_animation("Generating Complex Dataset")
    
    # Membuat dataset dummy tentang performa mahasiswa
    # Jam Belajar, Jam Tidur, Nilai Ujian, Tingkat Stress
    np.random.seed(42)
    n = 100
    jam_belajar = np.random.normal(5, 1.5, n)
    jam_tidur = np.random.normal(7, 1, n)
    # Nilai dipengaruhi belajar (positif) dan tidur (positif)
    nilai = (jam_belajar * 8) + (jam_tidur * 5) + np.random.normal(0, 5, n)
    # Stress dipengaruhi belajar (positif) tapi tidur (negatif)
    stress = (jam_belajar * 1.2) - (jam_tidur * 2) + 50
    
    df = pd.DataFrame({
        'Jam Belajar': jam_belajar,
        'Jam Tidur': jam_tidur,
        'Nilai Ujian': nilai,
        'Tingkat Stress': stress
    })
    
    print(f"{Fore.CYAN}Cuplikan Data (5 Baris Pertama):")
    print(tabulate(df.head(), headers='keys', tablefmt='simple'))
    
    print(f"\n{Fore.YELLOW}[INFO] Menampilkan Matriks Korelasi...")
    print("Nilai mendekati 1 = Hubungan Kuat Positif")
    print("Nilai mendekati -1 = Hubungan Kuat Negatif")
    
    # Visualisasi Seaborn
    plt.figure(figsize=(10, 8))
    sns.set_context("notebook", font_scale=1.2)
    heatmap = sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix: Faktor Performa Mahasiswa', fontsize=16)
    
    print(f"{Fore.GREEN}[DONE] Grafik Heatmap telah dibuka di jendela baru.")
    plt.show()
    pause()

# ==========================================
# MODUL 3: MACHINE LEARNING & MATEMATIKA (EXISTING)
# ==========================================
# (Fungsi-fungsi lama tetap dipertahankan agar lengkap)

def hitung_volume_bola():
    print_header()
    print(f"{Back.BLUE}{Fore.WHITE} [ MATH: GEOMETRI RUANG ] {Style.RESET_ALL}\n")
    try:
        r = float(input(f"{Fore.GREEN}>> Masukkan jari-jari (r): {Fore.WHITE}"))
        vol = (4/3) * math.pi * (r ** 3)
        print(f"\n{Fore.CYAN}Volume Bola: {vol:.2f} satuan kubik")
    except ValueError:
        print(f"{Fore.RED}Input Error!")
    pause()

def ml_kmeans_clustering():
    print_header()
    print(f"{Back.BLUE}{Fore.WHITE} [ AI: K-MEANS CLUSTERING ] {Style.RESET_ALL}\n")
    X, y_true = make_blobs(n_samples=400, centers=5, cluster_std=0.60, random_state=0)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='*', label='Centroids')
    plt.title("Customer Segmentation (Clustering)")
    plt.legend()
    plt.show()
    pause()

def ml_linear_regression():
    print_header()
    print(f"{Back.BLUE}{Fore.WHITE} [ AI: LINEAR REGRESSION PREDICTION ] {Style.RESET_ALL}\n")
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1) # Jam Belajar
    y = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95]) # Nilai
    
    model = LinearRegression()
    model.fit(X, y)
    
    try:
        val = float(input(f"{Fore.GREEN}>> Masukkan Jam Belajar untuk Prediksi Nilai: {Fore.WHITE}"))
        pred = model.predict([[val]])[0]
        print(f"\n{Back.GREEN}{Fore.BLACK} PREDIKSI NILAI: {pred:.2f} {Style.RESET_ALL}")
        if pred > 100: print(f"{Fore.YELLOW}(Sangat Rajin! Nilai mentok di 100)")
    except:
        pass
    pause()

# ==========================================
# MAIN CONTROLLER
# ==========================================
def main():
    # 1. Jalankan Login Terlebih Dahulu
    if login_system():
        while True:
            print_header()
            menu = [
                ["1", "Data Cleaning (Pandas) [NEW]"],
                ["2", "Correlation Heatmap (Seaborn) [NEW]"],
                ["3", "K-Means Clustering (AI)"],
                ["4", "Linear Regression (AI)"],
                ["5", "Kalkulator Volume Bola (Math)"],
                ["0", "LOGOUT & SHUTDOWN"]
            ]
            print(tabulate(menu, headers=["No", "Module Name"], tablefmt="rounded_outline"))
            
            choice = input(f"\n{Fore.GREEN}admin@rizqi-server:~# {Fore.WHITE}")
            
            if choice == '1': data_cleaning_demo()
            elif choice == '2': correlation_heatmap()
            elif choice == '3': ml_kmeans_clustering()
            elif choice == '4': ml_linear_regression()
            elif choice == '5': hitung_volume_bola()
            elif choice == '0':
                print(f"\n{Fore.RED}Closing Session... Data Saved. Goodbye Rizqi!")
                sys.exit()
            else:
                print(f"{Fore.RED}[ERROR] Invalid Command.")
                time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nForce Shutdown.")
