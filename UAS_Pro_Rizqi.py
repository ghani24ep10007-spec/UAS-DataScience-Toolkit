# ==========================================
# GABUNGAN MATERI MACHINE LEARNING + GAMBAR
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import warnings

# Abaikan warning agar output bersih
warnings.filterwarnings('ignore')
plt.style.use('ggplot') # Biar gambar lebih bagus

# ==========================================
# 1. LINEAR REGRESSION (Garis Lurus)
# ==========================================
print("\n--- 1. GENERATING LINEAR REGRESSION PLOT ---")
x_lin = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y_lin = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x_lin, y_lin)

def myfunc(x):
    return slope * x + intercept

mymodel_lin = list(map(myfunc, x_lin))

plt.figure(figsize=(12, 10)) # Ukuran kanvas besar untuk menampung semua gambar

# Plot 1: Linear
plt.subplot(2, 2, 1)
plt.scatter(x_lin, y_lin, color='blue', label='Data Asli')
plt.plot(x_lin, mymodel_lin, color='red', label=f'Regresi (R={r:.2f})')
plt.title('1. Linear Regression')
plt.legend()

# ==========================================
# 2. POLYNOMIAL REGRESSION (Garis Lengkung)
# ==========================================
print("--- 2. GENERATING POLYNOMIAL REGRESSION PLOT ---")
x_poly = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y_poly = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel_poly = np.poly1d(np.polyfit(x_poly, y_poly, 3))
myline = np.linspace(1, 22, 100)

# Plot 2: Polynomial
plt.subplot(2, 2, 2)
plt.scatter(x_poly, y_poly, color='green', label='Data Asli')
plt.plot(myline, mymodel_poly(myline), color='orange', linewidth=3, label='Poly Model')
plt.title('2. Polynomial Regression')
plt.legend()

# ==========================================
# 3. K-NEAREST NEIGHBORS (Grafik Akurasi/Elbow)
# ==========================================
print("--- 3. GENERATING KNN PLOT ---")
# Load data Iris
irisData = load_iris()
X_knn = irisData.data
y_knn = irisData.target
X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

neighbors = np.arange(1, 15)
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Plot 3: KNN
plt.subplot(2, 2, 3)
plt.plot(neighbors, test_accuracy, label='Testing Accuracy', marker='o', color='purple')
plt.title('3. KNN: Mencari k Terbaik')
plt.xlabel('Nilai k')
plt.ylabel('Akurasi')
plt.legend()

# ==========================================
# 4. LOGISTIC REGRESSION (Peta Merah/Hijau)
# ==========================================
print("--- 4. GENERATING LOGISTIC REGRESSION PLOT ---")

# Buat Data Dummy dulu (Biar tidak error file not found)
data_dummy = {
    'Age': np.random.randint(18, 60, 400),
    'EstimatedSalary': np.random.randint(15000, 150000, 400),
    'Purchased': np.random.choice([0, 1], 400)
}
dataset = pd.DataFrame(data_dummy)

X_log = dataset.iloc[:, [0, 1]].values
y_log = dataset.iloc[:, 2].values

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train_log = sc.fit_transform(X_train_log)
X_test_log = sc.transform(X_test_log)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_log, y_train_log)

# Visualisasi
X_set, y_set = X_test_log, y_test_log
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.05),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.05)
)

# Plot 4: Logistic
plt.subplot(2, 2, 4)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.3, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

colors = ['red', 'green']
labels = ['Not Purchased', 'Purchased']
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=colors[i], label=labels[j], edgecolor='black', s=30)

plt.title('4. Logistic Regression')
plt.legend()

plt.tight_layout() # Agar gambar tidak saling tabrakan
plt.show()

print("✅ SUKSES! Semua gambar telah berhasil dibuat.")