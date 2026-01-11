import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_iris

# ==========================================
# KONFIGURASI TAMPILAN & POP-UP
# ==========================================
def clear_screen():
    """Membersihkan layar terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_popup(title, message, type="info"):
    """
    Menampilkan Pop-up Window agar presentasi lebih interaktif.
    type: 'info', 'warning', 'error'
    """
    # Membuat root window tersembunyi (agar tidak muncul jendela kosong)
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) # Agar pop-up muncul paling depan
    
    if type == "info":
        messagebox.showinfo(title, message)
    elif type == "warning":
        messagebox.showwarning(title, message)
    elif type == "error":
        messagebox.showerror(title, message)
        
    root.destroy()

def print_header():
    """Menampilkan Header Identitas Mahasiswa."""
    clear_screen()
    print("="*75)
    print("      INTEGRATED DATA SCIENCE & MATHEMATICS TOOLKIT (UAS IMK) 🧮🤖")
    print("="*75)
    print(" Nama  : Rizqi Ghani Adinata")
    print(" Prodi : Sistem Informasi (Semester 3)")
    print(" Dosen : Pak Abdul Haq")
    print("-" * 75)

def pause():
    """Memberi jeda agar user bisa membaca output."""
    input("\n[INFO] Tekan Enter untuk kembali ke menu utama...")

# ==========================================
# BAGIAN 1: MATEMATIKA DASAR
# ==========================================

def hitung_volume_bola():
    print_header()
    print("[ MATEMATIKA ] Menghitung Volume Bola")
    print("=" * 40)
    
    while True:
        try:
            r_input = input("Masukkan jari-jari bola (atau ketik 'x' untuk keluar): ")
            if r_input.lower() == 'x': 
                break
            
            jari_jari = float(r_input)
            
            if jari_jari <= 0:
                show_popup("Input Error", "Jari-jari harus lebih besar dari 0!", "error")
                print("Error: Jari-jari harus lebih besar dari 0!")
                continue
            
            volume = (4/3) * math.pi * (jari_jari ** 3)
            
            print(f"\nJari-jari: {jari_jari}")
            print(f"Volume bola: {volume:.2f}")
            print("-" * 40)
            
            lagi = input("Hitung lagi? (y/n): ").lower()
            if lagi != 'y':
                break
                
        except ValueError:
            show_popup("Input Error", "Masukkan angka yang valid!", "error")
            print("Error: Masukkan angka yang valid!")
    pause()

def akar_persamaan_kuadrat():
    print_header()
    print("[ MATEMATIKA ] Akar Persamaan Kuadrat (ax^2 + bx + c = 0)")
    try:
        a = float(input("Masukkan nilai a: "))
        b = float(input("Masukkan nilai b: "))
        c = float(input("Masukkan nilai c: "))
        
        D = b**2 - 4*a*c 
        print(f"\nDiskriminan (D) = {D}")
        
        if D > 0:
            x1 = (-b + np.sqrt(D)) / (2*a)
            x2 = (-b - np.sqrt(D)) / (2*a)
            result_msg = f"Dua Akar Real Berbeda:\nx1 = {x1:.2f}\nx2 = {x2:.2f}"
        elif D == 0:
            x = -b / (2*a)
            result_msg = f"Akar Kembar:\nx = {x:.2f}"
        else:
            result_msg = "Akar Imajiner (Tidak ada solusi real)"
            
        print(f"\nHasil:\n{result_msg}")
        show_popup("Hasil Perhitungan", result_msg, "info")
        
    except ValueError:
        print("Input error! Harap masukkan angka.")
    pause()

def get_matrix_determinant(matrix):
    n = len(matrix)
    if n == 1: return matrix[0][0]
    if n == 2: return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    
    det = 0
    for c in range(n):
        sub_matrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1)**c) * matrix[0][c] * get_matrix_determinant(sub_matrix)
    return det

def menu_determinan():
    print_header()
    print("[ MATEMATIKA ] Determinan Matriks (Rekursif)")
    try:
        n = int(input("Masukkan ordo matriks (2, 3, atau 4): "))
        if n < 2:
            show_popup("Peringatan", "Ordo minimal matriks adalah 2!", "warning")
            return

        print(f"Masukkan elemen matriks {n}x{n}:")
        matrix = []
        for i in range(n):
            row_input = input(f"Baris {i+1} (pisahkan angka dengan spasi): ")
            row = list(map(float, row_input.split()))
            if len(row) != n:
                print(f"Error: Harap masukkan tepat {n} angka!")
                return
            matrix.append(row)
        
        print("\nSedang menghitung...")
        det = get_matrix_determinant(matrix)
        print(f"\nDeterminannya adalah: {det}")
        show_popup("Hasil Determinan", f"Nilai Determinan Matriks: {det}", "info")
        
    except Exception as e:
        print(f"Error: {e}")
    pause()

# ==========================================
# BAGIAN 2: MACHINE LEARNING
# ==========================================

def demo_linear_regression_scipy():
    print_header()
    print("[ ML ] Linear Regression (Metode Scipy Stats)")
    
    x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
    y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
    
    print(f"Data X (Input) : {x}")
    print(f"Data Y (Target): {y}")
    
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    
    def myfunc(x):
        return slope * x + intercept
    
    mymodel = list(map(myfunc, x))
    
    print(f"\nSlope: {slope:.4f} | Intercept: {intercept:.4f}")
    
    show_popup("Visualisasi Data", "Grafik Linear Regression akan ditampilkan setelah ini.", "info")
    
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label='Data Asli')
    plt.plot(x, mymodel, color='red', label='Garis Regresi')
    plt.title('Linear Regression (Scipy Implementation)')
    plt.legend()
    plt.grid(True)
    plt.show()
    pause()

def demo_polynomial_regression():
    print_header()
    print("[ ML ] Polynomial Regression (Data Non-Linear)")
    
    x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
    y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
    
    print("Membuat model Polynomial Degree 3...")
    
    mymodel = np.poly1d(np.polyfit(x, y, 3)) 
    myline = np.linspace(1, 22, 100)
    
    show_popup("Visualisasi Data", "Grafik Polynomial Regression (Kurva) akan ditampilkan.", "info")
    
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label='Data Aktual')
    plt.plot(myline, mymodel(myline), color='green', label='Polyfit Degree 3')
    plt.title('Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    pause()

def demo_knn_iris():
    print_header()
    print("[ ML ] K-Nearest Neighbors (Dataset Iris)")
    
    irisData = load_iris()
    X = irisData.data
    y = irisData.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    
    print("1. Melakukan Prediksi Single dengan K=7...")
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print(f"   Akurasi Model: {acc:.2f}")
    
    show_popup("Analisis K-Neighbor", "Selanjutnya: Grafik perbandingan akurasi berdasarkan jumlah K.", "info")
    
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)

    plt.figure(figsize=(10, 6))
    plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Jumlah Neighbor (K)')
    plt.ylabel('Akurasi')
    plt.title('KNN: Analisis Jumlah Tetangga Terbaik')
    plt.grid(True)
    plt.show()
    pause()

def demo_logistic_regression():
    print_header()
    print("[ ML ] Logistic Regression (Klasifikasi Iklan Medsos)")
    
    if os.path.exists('Social_Network_Ads.csv'):
        try:
            dataset = pd.read_csv('Social_Network_Ads.csv')
            X = dataset.iloc[:, [2, 3]].values
            y = dataset.iloc[:, 4].values
            print("Status: Menggunakan file 'Social_Network_Ads.csv'.")
        except:
            print("Status: File error, menggunakan Dummy Data Generator.")
            X = np.random.rand(100, 2) * 50 + 20 
            y = np.random.randint(0, 2, 100)
    else:
        print("Status: File CSV tidak ditemukan. Menggunakan Dummy Data Generator.")
        X = np.random.rand(100, 2) * 50 + 20
        y = np.random.randint(0, 2, 100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix Result:")
    print(cm)
    
    show_popup("Visualisasi Klasifikasi", "Menampilkan Contour Plot untuk Logistic Regression.", "info")
    
    try:
        plt.figure(figsize=(10,6))
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j, edgecolors='black')
                        
        plt.title('Logistic Regression (Test Set)')
        plt.xlabel('Age (Scaled)')
        plt.ylabel('Estimated Salary (Scaled)')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Gagal menampilkan grafik: {e}")
    
    pause()

# ==========================================
# MENU UTAMA
# ==========================================
def main():
    # Pop-up Selamat Datang
    show_popup("Selamat Datang", "Presentasi UAS Machine Learning & Matematika\nOleh: Rizqi Ghani Adinata\nUntuk: Pak Abdul Haq", "info")
    
    while True:
        print_header()
        print("PILIH MODUL PRESENTASI:")
        print("1. [Math] Volume Bola (Looping & Validasi)")
        print("2. [Math] Akar Persamaan Kuadrat")
        print("3. [Math] Determinan Matriks (Rekursif)")
        print("-" * 40)
        print("4. [ML]   Linear Regression (Scipy Stats)")
        print("5. [ML]   Polynomial Regression (Numpy)")
        print("6. [ML]   KNN Clustering (Iris Dataset)")
        print("7. [ML]   Logistic Regression (Classification)")
        print("-" * 40)
        print("0. KELUAR")
        
        choice = input("\nPilihan Anda (0-7): ")
        
        if choice == '1': hitung_volume_bola()
        elif choice == '2': akar_persamaan_kuadrat()
        elif choice == '3': menu_determinan()
        elif choice == '4': demo_linear_regression_scipy()
        elif choice == '5': demo_polynomial_regression()
        elif choice == '6': demo_knn_iris()
        elif choice == '7': demo_logistic_regression()
        elif choice == '0':
            # Pop-up Perpisahan
            show_popup("Terima Kasih", "Terima kasih Pak Abdul Haq atas bimbingannya.\nPresentasi Selesai.", "info")
            sys.exit()
        else:
            show_popup("Pilihan Salah", "Mohon masukkan angka 0 sampai 7.", "warning")
            pause()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram dihentikan pengguna.")