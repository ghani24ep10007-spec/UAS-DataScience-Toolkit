import os
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris

# --- LIBRARY TAMBAHAN UNTUK TAMPILAN KEREN ---
from colorama import init, Fore, Style, Back
from tabulate import tabulate

# Inisialisasi Colorama (Wajib untuk Windows)
init(autoreset=True)

# ==========================================
# FUNGSI KOSMETIK (AGAR TAMPILAN MEWAH)
# ==========================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def loading_animation(text="Memproses Data", duration=2):
    """Membuat efek loading bar agar terlihat canggih"""
    print(f"\n{Fore.YELLOW}{text}...", end="")
    for i in range(20):
        time.sleep(duration / 20)
        sys.stdout.write(f"{Fore.GREEN}█")
        sys.stdout.flush()
    print(f" {Fore.CYAN}[SELESAI]\n")
    time.sleep(0.5)

def print_header():
    clear_screen()
    # ASCII ART NAMA ANDA
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + """
  ____  _         _    ____  _                 _ 
 |  _ \(_)_______(_)  / ___|| |__   __ _ _ __ (_)
 | |_) | |_  / __| | | |  _| '_ \ / _` | '_ \| |
 |  _ <| |/ / (__| | | |_| | | | | (_| | | | | |
 |_| \_\_/___\___|_|  \____|_| |_|\__,_|_| |_|_|
                                                 
    DATA SCIENCE & MATH TOOLKIT v2.0
    """)
    print(Fore.CYAN + "="*80)
    
    # Informasi Mahasiswa dalam Tabel
    info = [
        ["Nama Developer", "Rizqi Ghani Adinata"],
        ["NIM / Prodi", "Sistem Informasi (Semester 3)"],
        ["Dosen Pengampu", "Bapak Abdul Haq"],
        ["Project", "UAS Interaksi Manusia & Komputer"]
    ]
    print(tabulate(info, tablefmt="fancy_grid", stralign="left"))
    print(Fore.YELLOW + "\n[SYSTEM READY] Menunggu input user...\n")

def pause():
    input(f"\n{Back.BLUE}{Fore.WHITE} TEKAN ENTER UNTUK KEMBALI {Style.RESET_ALL}")

# ==========================================
# MODUL 1: MATEMATIKA
# ==========================================
def hitung_volume_bola():
    print_header()
    print(f"{Back.MAGENTA}{Fore.WHITE} [ MODUL MATEMATIKA: VOLUME BOLA ] {Style.RESET_ALL}\n")
    
    while True:
        try:
            r_input = input(f"{Fore.GREEN}>> Masukkan jari-jari bola (ketik 'x' untuk keluar): {Fore.WHITE}")
            if r_input.lower() == 'x': break
            
            r = float(r_input)
            if r <= 0:
                print(f"{Fore.RED}[ERROR] Jari-jari harus positif!")
                continue
            
            loading_animation("Menghitung Rumus V = 4/3 π r³")
            vol = (4/3) * math.pi * (r ** 3)
            
            # Tampilkan hasil dengan tabel
            hasil = [["Jari-jari (r)", f"{r}"], ["Volume Bola", f"{vol:.2f}"]]
            print(tabulate(hasil, headers=["Parameter", "Nilai"], tablefmt="grid"))
            
            if input(f"\n{Fore.YELLOW}Hitung lagi? (y/n): ").lower() != 'y': break
        except ValueError:
            print(f"{Fore.RED}[ERROR] Masukkan angka saja!")
    pause()

def akar_persamaan_kuadrat():
    print_header()
    print(f"{Back.MAGENTA}{Fore.WHITE} [ MODUL MATEMATIKA: PERSAMAAN KUADRAT ] {Style.RESET_ALL}\n")
    try:
        a = float(input("Nilai a: "))
        b = float(input("Nilai b: "))
        c = float(input("Nilai c: "))
        
        loading_animation("Mencari Akar-akar")
        
        D = b**2 - 4*a*c
        data = [["Diskriminan (D)", f"{D}"]]
        
        if D > 0:
            x1 = (-b + np.sqrt(D)) / (2*a)
            x2 = (-b - np.sqrt(D)) / (2*a)
            data.append(["Status", "Dua Akar Real Berbeda"])
            data.append(["x1", f"{x1:.2f}"])
            data.append(["x2", f"{x2:.2f}"])
        elif D == 0:
            x = -b / (2*a)
            data.append(["Status", "Akar Kembar"])
            data.append(["x", f"{x:.2f}"])
        else:
            data.append(["Status", "Akar Imajiner (Kompleks)"])
            
        print(tabulate(data, tablefmt="fancy_grid"))
        
    except ValueError:
        print(f"{Fore.RED}Input error!")
    pause()

def menu_determinan():
    print_header()
    print(f"{Back.MAGENTA}{Fore.WHITE} [ MODUL MATEMATIKA: DETERMINAN MATRIKS ] {Style.RESET_ALL}\n")
    
    # Fungsi Determinan tetap sama
    def get_det(matrix):
        n = len(matrix)
        if n == 1: return matrix[0][0]
        if n == 2: return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
        det = 0
        for c in range(n):
            sub = [row[:c] + row[c+1:] for row in matrix[1:]]
            det += ((-1)**c) * matrix[0][c] * get_det(sub)
        return det

    try:
        n = int(input(f"{Fore.GREEN}Masukkan ordo matriks (2-4): {Fore.WHITE}"))
        print(f"{Fore.CYAN}Masukkan elemen matriks per baris:")
        matrix = []
        for i in range(n):
            row = list(map(float, input(f"Baris {i+1}: ").split()))
            matrix.append(row)
            
        loading_animation("Menghitung Determinan Secara Rekursif")
        det = get_det(matrix)
        
        print(f"\n{Back.GREEN}{Fore.BLACK} HASIL AKHIR: {det} {Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Error: {e}")
    pause()

# ==========================================
# MODUL 2: MACHINE LEARNING
# ==========================================
def demo_linear_regression():
    print_header()
    print(f"{Back.BLUE}{Fore.WHITE} [ MACHINE LEARNING: LINEAR REGRESSION ] {Style.RESET_ALL}\n")
    
    x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
    y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
    
    loading_animation("Training Model AI")
    
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    
    # Tampilkan Statistik Model
    stats_data = [
        ["Slope (Kemiringan)", f"{slope:.4f}"],
        ["Intercept", f"{intercept:.4f}"],
        ["Akurasi (R²)", f"{r**2:.4f}"]
    ]
    print(tabulate(stats_data, headers=["Metric", "Value"], tablefmt="heavy_outline"))
    print(f"\n{Fore.YELLOW}[INFO] Grafik akan muncul di jendela baru...")
    
    def myfunc(x): return slope * x + intercept
    mymodel = list(map(myfunc, x))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data Asli')
    plt.plot(x, mymodel, color='red', linewidth=2, label='Garis Prediksi AI')
    plt.title(f'Linear Regression (Akurasi: {r**2:.2f})')
    plt.xlabel('Variabel X')
    plt.ylabel('Variabel Y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    pause()

def demo_knn():
    print_header()
    print(f"{Back.BLUE}{Fore.WHITE} [ MACHINE LEARNING: KNN CLASSIFICATION ] {Style.RESET_ALL}\n")
    
    loading_animation("Memuat Dataset Bunga Iris")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    loading_animation("Melatih Model Tetangga Terdekat (K=7)")
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    
    print(f"\n{Back.GREEN}{Fore.BLACK} MODEL SUKSES DILATIH! {Style.RESET_ALL}")
    print(f"Akurasi Prediksi: {Fore.GREEN}{acc*100:.2f}%")
    
    print(f"\n{Fore.YELLOW}[INFO] Menampilkan grafik analisis K...")
    neighbors = np.arange(1, 9)
    test_acc = []
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        test_acc.append(knn.score(X_test, y_test))
        
    plt.figure(figsize=(10, 6))
    plt.plot(neighbors, test_acc, marker='o', linestyle='-', color='purple')
    plt.title('Analisis Akurasi Berdasarkan Jumlah Tetangga (K)')
    plt.xlabel('Nilai K')
    plt.ylabel('Akurasi')
    plt.grid(True)
    plt.show()
    pause()

def demo_logistic():
    print_header()
    print(f"{Back.BLUE}{Fore.WHITE} [ MACHINE LEARNING: LOGISTIC REGRESSION ] {Style.RESET_ALL}\n")
    
    # Cek CSV
    if os.path.exists('Social_Network_Ads.csv'):
        print(f"{Fore.GREEN}[OK] File Dataset Ditemukan.")
        df = pd.read_csv('Social_Network_Ads.csv')
        X = df.iloc[:, [2, 3]].values
        y = df.iloc[:, 4].values
    else:
        print(f"{Fore.RED}[WARNING] File CSV tidak ada. Menggunakan Dummy Data.")
        X = np.random.rand(100, 2) * 50
        y = np.random.randint(0, 2, 100)

    loading_animation("Membangun Decision Boundary")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    
    # Visualisasi
    print(f"{Fore.YELLOW}[INFO] Menampilkan Peta Klasifikasi...")
    plt.figure(figsize=(10, 6))
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min()-1, X_set[:, 0].max()+1, 0.01),
                         np.arange(X_set[:, 1].min()-1, X_set[:, 1].max()+1, 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j, edgecolors='black')
    plt.title('Logistic Regression (Klasifikasi User)')
    plt.legend()
    plt.show()
    pause()

# ==========================================
# MAIN MENU
# ==========================================
def main():
    while True:
        print_header()
        menu = [
            ["1", "Hitung Volume Bola"],
            ["2", "Akar Persamaan Kuadrat"],
            ["3", "Determinan Matriks"],
            ["4", "Linear Regression (Prediksi)"],
            ["5", "KNN Classification (Iris)"],
            ["6", "Logistic Regression (Ads)"],
            ["0", "KELUAR"]
        ]
        print(tabulate(menu, headers=["No", "Pilihan Modul"], tablefmt="rounded_grid"))
        
        choice = input(f"\n{Fore.GREEN}>> Masukkan Pilihan (0-6): {Fore.WHITE}")
        
        if choice == '1': hitung_volume_bola()
        elif choice == '2': akar_persamaan_kuadrat()
        elif choice == '3': menu_determinan()
        elif choice == '4': demo_linear_regression()
        elif choice == '5': demo_knn()
        elif choice == '6': demo_logistic()
        elif choice == '0':
            print(f"\n{Fore.CYAN}Terima kasih Rizqi. Sukses Presentasinya!{Style.RESET_ALL}")
            sys.exit()
        else:
            print(f"{Fore.RED}Pilihan tidak valid!")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram stopped.")
