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
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_iris, make_blobs

# --- LIBRARY KOSMETIK CLI ---
from colorama import init, Fore, Style, Back
from tabulate import tabulate

# Inisialisasi Colorama
init(autoreset=True)

# ==========================================
# KONFIGURASI SISTEM & UI
# ==========================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def loading_animation(text="Memproses Data", duration=1.5):
    """Efek loading bar ala terminal"""
    print(f"\n{Fore.YELLOW}{text}...", end="")
    width = 30
    for i in range(width):
        time.sleep(duration / width)
        sys.stdout.write(f"{Fore.GREEN}█")
        sys.stdout.flush()
    print(f" {Fore.CYAN}[OK]\n")
    time.sleep(0.3)

def print_header():
    clear_screen()
    print(Fore.GREEN + Style.BRIGHT + "="*85)
    print(Fore.GREEN + r"""
  ____       _          ____       _                         __  __       _   _     
 |  _ \ __ _| |_ __ _  / ___|  ___(_) ___ _ __   ___ ___    |  \/  | __ _| |_| |__  
 | | | / _` | __/ _` | \___ \ / __| |/ _ \ '_ \ / __/ _ \   | |\/| |/ _` | __| '_ \ 
 | |_| | (_| | || (_| |  ___) | (__| |  __/ | | | (_|  __/  | |  | | (_| | |_| | | |
 |____/ \__,_|\__\__,_| |____/ \___|_|\___|_| |_|\___\___|  |_|  |_|\__,_|\__|_| |_|
                                                                                    
    INTEGRATED TOOLKIT v2.5 - UAS Semester 3
    """)
    print(Fore.GREEN + "="*85)
    
    info = [
        ["Developer", "Rizqi Ghani Adinata"],
        ["NIM / Prodi", "Sistem Informasi (Sem. 3)"],
        ["Mata Kuliah", "Interaksi Manusia & Komputer"],
        ["Dosen Pengampu", "Bapak Abdul Haq"],
        ["Status Sistem", "ONLINE - ACCESS GRANTED"]
    ]
    print(tabulate(info, tablefmt="simple_grid", stralign="left"))
    print(Fore.YELLOW + "\n[COMMAND CENTER] Menunggu instruksi user...\n")

def pause():
    input(f"\n{Back.GREEN}{Fore.BLACK} ENTER {Style.RESET_ALL} {Fore.WHITE}untuk kembali ke menu utama...")

# ==========================================
# MODUL 1: MATEMATIKA MURNI & STATISTIKA
# ==========================================
def hitung_volume_bola():
    print_header()
    print(f"{Back.MAGENTA}{Fore.WHITE} [ MODUL: GEOMETRI RUANG ] {Style.RESET_ALL}\n")
    try:
        r = float(input(f"{Fore.GREEN}>> Masukkan jari-jari bola (r): {Fore.WHITE}"))
        loading_animation("Mengkalkulasi V = 4/3 π r³")
        vol = (4/3) * math.pi * (r ** 3)
        area = 4 * math.pi * (r ** 2)
        
        data = [
            ["Jari-jari (r)", f"{r}"],
            ["Volume", f"{vol:.4f}"],
            ["Luas Permukaan", f"{area:.4f}"]
        ]
        print(tabulate(data, headers=["Parameter", "Nilai"], tablefmt="fancy_grid"))
    except ValueError:
        print(f"{Fore.RED}[ERROR] Input harus berupa angka!")
    pause()

def operasi_matriks():
    print_header()
    print(f"{Back.MAGENTA}{Fore.WHITE} [ MODUL: OPERASI MATRIKS ] {Style.RESET_ALL}\n")
    print("1. Determinan Matriks")
    print("2. Perkalian Matriks (2x2)")
    pilih = input(f"\n{Fore.GREEN}>> Pilih Sub-menu (1/2): {Fore.WHITE}")
    
    if pilih == '1':
        try:
            n = int(input("Ordo matriks (2/3): "))
            print(f"{Fore.CYAN}Input elemen baris demi baris (pisahkan dengan spasi):")
            mat = []
            for i in range(n):
                row = list(map(float, input(f"Baris {i+1}: ").split()))
                if len(row) != n: raise ValueError("Jumlah kolom tidak sesuai ordo")
                mat.append(row)
            
            loading_animation("Menghitung Determinan")
            det = np.linalg.det(mat)
            print(f"\n{Back.BLUE}{Fore.WHITE} DETERMINAN: {det:.2f} {Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error: {e}")

    elif pilih == '2':
        print(f"{Fore.CYAN}Matriks A (2x2):")
        a = [list(map(float, input("Baris 1: ").split())), list(map(float, input("Baris 2: ").split()))]
        print(f"{Fore.CYAN}Matriks B (2x2):")
        b = [list(map(float, input("Baris 1: ").split())), list(map(float, input("Baris 2: ").split()))]
        
        loading_animation("Mengalikan A x B")
        res = np.dot(a, b)
        print(f"\n{Fore.YELLOW}Hasil Perkalian:")
        print(tabulate(res, tablefmt="grid"))
    pause()

def statistika_dasar():
    print_header()
    print(f"{Back.MAGENTA}{Fore.WHITE} [ MODUL: STATISTIKA DESKRIPTIF ] {Style.RESET_ALL}\n")
    try:
        raw = input(f"{Fore.GREEN}>> Masukkan data angka (pisahkan koma): {Fore.WHITE}")
        data = [float(x) for x in raw.split(',')]
        
        loading_animation("Menganalisis Distribusi Data")
        
        stats_info = [
            ["Mean (Rata-rata)", np.mean(data)],
            ["Median (Nilai Tengah)", np.median(data)],
            ["Modus (Sering Muncul)", stats.mode(data)[0][0] if len(data) > 1 else "N/A"],
            ["Standar Deviasi", f"{np.std(data):.4f}"],
            ["Varians", f"{np.var(data):.4f}"],
            ["Nilai Max", np.max(data)],
            ["Nilai Min", np.min(data)]
        ]
        print(tabulate(stats_info, headers=["Metrik", "Nilai"], tablefmt="heavy_outline"))
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Format data salah. Contoh: 10,20,30,40")
    pause()

# ==========================================
# MODUL 2: ARTIFICIAL INTELLIGENCE & ML
# ==========================================
def ml_linear_regression():
    print_header()
    print(f"{Back.BLUE}{Fore.WHITE} [ SUPERVISED ML: LINEAR REGRESSION ] {Style.RESET_ALL}\n")
    
    # Data Dummy: Pengalaman Kerja (Thn) vs Gaji (Juta)
    X = np.array([1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.9, 4.0]).reshape(-1, 1)
    y = np.array([4, 4.2, 4.5, 5, 5.2, 6, 6.2, 6.4, 7, 7.2])
    
    loading_animation("Training Model (Salary Prediction)")
    model = LinearRegression()
    model.fit(X, y)
    
    # Prediksi User
    try:
        exp = float(input(f"\n{Fore.GREEN}>> Prediksi Gaji untuk Pengalaman (tahun): {Fore.WHITE}"))
        pred = model.predict([[exp]])[0]
        print(f"\n{Back.GREEN}{Fore.BLACK} PREDIKSI GAJI: Rp {pred:.2f} Juta {Style.RESET_ALL}")
    except:
        pass

    # Visualisasi
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Data Real')
    plt.plot(X, model.predict(X), color='blue', label='Regression Line')
    plt.title('Salary vs Experience (Linear Regression)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary (Millions IDR)')
    plt.legend()
    plt.grid(True)
    print(f"{Fore.YELLOW}\n[INFO] Menutup grafik untuk lanjut...")
    plt.show()
    pause()

def ml_kmeans_clustering():
    print_header()
    print(f"{Back.BLUE}{Fore.WHITE} [ UNSUPERVISED ML: K-MEANS CLUSTERING ] {Style.RESET_ALL}\n")
    
    loading_animation("Generating Data Dummy (300 points)")
    # Membuat 300 data random yang terpusat di 4 titik (centers)
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    loading_animation("Fitting K-Means Algorithm")
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    print(f"{Fore.CYAN}Model selesai mengelompokkan data menjadi 4 Cluster.")
    print(f"Center Clusters Coordinates:\n{kmeans.cluster_centers_}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X', label='Centroids')
    plt.title("K-Means Clustering Visualization")
    plt.legend()
    plt.show()
    pause()

def ml_logistic_dashboard():
    print_header()
    print(f"{Back.BLUE}{Fore.WHITE} [ SUPERVISED ML: LOGISTIC REGRESSION ] {Style.RESET_ALL}\n")
    
    # Cek Dataset Real atau Dummy
    if os.path.exists('Social_Network_Ads.csv'):
        print(f"{Fore.GREEN}[DATASET] Menggunakan Social_Network_Ads.csv")
        df = pd.read_csv('Social_Network_Ads.csv')
        X = df.iloc[:, [2, 3]].values
        y = df.iloc[:, 4].values
    else:
        print(f"{Fore.YELLOW}[WARNING] File CSV tidak ada. Menggunakan Generated Data.")
        X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=2.5)
        # Normalisasi agar mirip data umur/gaji
        X = np.abs(X * 10) 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{Back.GREEN}{Fore.BLACK} AKURASI MODEL: {acc*100:.2f}% {Style.RESET_ALL}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Visualisasi Contour
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
                    c=ListedColormap(('red', 'green'))(i), label=f'Class {j}', edgecolors='black')
    plt.title('Logistic Regression (Classification Boundary)')
    plt.legend()
    plt.show()
    pause()

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    while True:
        print_header()
        menu = [
            ["1", "Hitung Geometri (Bola)"],
            ["2", "Operasi Matriks & Determinan"],
            ["3", "Statistika Deskriptif"],
            ["4", "Linear Regression (Prediksi Gaji)"],
            ["5", "Logistic Regression (Klasifikasi)"],
            ["6", "K-Means Clustering (Pengelompokan)"],
            ["0", "SHUTDOWN SYSTEM"]
        ]
        print(tabulate(menu, headers=["ID", "Modul Aplikasi"], tablefmt="rounded_outline"))
        
        choice = input(f"\n{Fore.GREEN}root@rizqi-server:~# {Fore.WHITE}")
        
        if choice == '1': hitung_volume_bola()
        elif choice == '2': operasi_matriks()
        elif choice == '3': statistika_dasar()
        elif choice == '4': ml_linear_regression()
        elif choice == '5': ml_logistic_dashboard()
        elif choice == '6': ml_kmeans_clustering()
        elif choice == '0':
            print(f"\n{Fore.CYAN}System Shutting Down... Goodbye Rizqi!{Style.RESET_ALL}")
            sys.exit()
        else:
            print(f"{Fore.RED}[ERROR] Command not found!")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nForce Close.")
