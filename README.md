# Laporan Proyek Machine Learning - Salma Oktarina
## Domain Proyek
### Latar Belakang:
Polycystic Ovary Syndrome (PCOS) adalah gangguan hormonal pada wanita yang ditandai dengan munculnya kista kecil berisi cairan pada ovarium. Penyebab pasti PCOS masih belum diketahui secara pasti, namun beberapa faktor yang berkontribusi termasuk resistensi insulin, kelebihan hormon androgen, dan siklus menstruasi tidak teratur. PCOS dapat mengganggu proses ovulasi dan produksi hormon estrogen sehingga dapat menyebabkan gangguan kesuburan.

Deteksi dini PCOS sangat penting untuk mencegah komplikasi serius seperti infertilitas, diabetes tipe 2, serta penyakit kardiovaskular. Namun, banyak wanita yang tidak menyadari adanya PCOS karena gejalanya sering dianggap biasa atau tidak kentara. Oleh karena itu, diperlukan sistem prediksi berbasis data medis yang dapat membantu mengidentifikasi kemungkinan PCOS secara dini.

### Masalah:
- PCOS mempengaruhi kualitas hidup dan kesuburan wanita usia produktif.
- Deteksi dini dapat mencegah komplikasi jangka panjang yang berbahaya.

### Solusi:
Sistem prediksi berbasis machine learning dapat membantu tenaga medis dalam diagnosis awal dengan data yang tersedia secara cepat dan akurat.

Referensi:  [1] Medula Veny Anisya, Ratna Dewi PS, Rizki Hanriko, Risti Graharti. "Polycystic Ovary Syndrome: Resiko Infertilitas yang Dapat Dicegah Melalui Penurunan Berat Badan Pada Wanita Obesitas", Jurnal Medula, 2019. [Online]. Available: https://www.journalofmedula.com/index.php/medula/article/view/268

## Business Understanding

### Problem Statements
- Bagaimana cara memanfaatkan data medis untuk mengklasifikasikan pasien yang berpotensi mengalami PCOS?
- Algoritma machine learning apa yang paling efektif untuk memprediksi kondisi PCOS berdasarkan data yang tersedia?

### Goals
- Mengembangkan model klasifikasi untuk mendeteksi kemungkinan PCOS dari data medis.
- Membandingkan performa model Logistic Regression dan Random Forest.

### Solution Statements
- Menggunakan Logistic Regression sebagai baseline model karena kecepatan dan kemudahannya dalam interpretasi hasil.
- Menggunakan Random Forest untuk menangani interaksi fitur non-linear dan potensi overfitting, serta meningkatkan akurasi prediksi.
- Melakukan hyperparameter tuning pada Random Forest untuk optimasi performa jika diperlukan.
- Metrik evaluasi yang digunakan untuk menilai performa model mencakup akurasi, precision, recall, dan F1-score agar sesuai dengan konteks data yang kemungkinan imbalance.

## Data Understanding
Dataset yang digunakan diambil dari Kaggle - Polycystic Ovary Syndrome (PCOS). Dataset ini berisi data demografi yang dibuat berdasarkan Kriteria Rotterdam. Kriteria Rotterdam, yang dibuat pada tahun 2003 oleh konsensus internasional para ahli, mendefinisikan bahwa, agar seorang wanita dapat didiagnosis dengan sindrom ini, ia harus memenuhi setidaknya dua dari tiga kriteria.

Sumber Data: https://www.kaggle.com/datasets/lucass0s0/polycystic-ovary-syndrome-pcos
Dataset ini terdiri dari 3000 sampel (baris data) dan 5 fitur (kolom), termasuk variabel target ('PCOS_Diagnosis), semua fitur bertipe numerik.

Variabel Utama yang Digunakan:
- Age: Umur
- BMI: Indeks massa tubuh
- Menstrual_Irregularity: Ketidakteraturan siklus menstruasi (0 tidak lancar dan 1 lancar)
- Testosterone_Level: Kadar hormon testosteron (ng/dL)
- Antral_Follicle_Count: Jumlah folikel antral berdasarkan USG
- PCOS_Diagnosis: Label target (0 jika tidak, 1 jika positif PCOS)

## Data Exploration
- Informasi Dasar & Statistik: Dataset awal meiliki 3000 entri dan 9 kolom tanpa nilai null.
- Duplikasi Data: Tidak ditemukan baris data yang duplikat.
- Distribusi Fitur: Visualisasi histogram menunjukkan bahwa distribusi data merata dan normal.
- Outliers: Box Plot menunjukkan ```Testosterone_Level``` memiliki nilai rentang lumayan jauh.
- Distribusi Target: Terdapat ketimpangan kelas, lebih banyak kelas tidak terindikasi PCOS (PCOS_Diagnosis = 0) sebanyak 2600 data dan terindikasi PCOS (PCOS_Diagnosis = 1) sebanyak 600 data.
```
PCOS Distribusi:
0    2400
1     600
Name: count, dtype: int64
```
![Screenshot (1519)](https://github.com/user-attachments/assets/dfdcbfdd-5d65-45a1-961f-c290ddbca99b)

- Korelasi Fitur:
-- Berdasarkan matriks korelasi, terlihat fitur yang paling berkorelasi positif terhadap kemungkina terindikasi PCOS (PCOS_Diagnosis) adalah jumlah folikel antral (0.87), kadar testosteron dan ketidakteraturan mestruasi (0.78), dan indeks massa tubuh (0.29).
-- Hal ini menunjukkan bahwa jumlah folikel antral, kadar testosteron, dan ketidakteraturan mestruasi berkaitan erat dengan adanya indikasi PCOS.
![Screenshot (1520)](https://github.com/user-attachments/assets/4967945c-0a0b-4582-98b5-66196fcbca74)

## Data Preparation
Tahapan yang Dilakukan:
- Penanganan Outliers (Metode IQR): Outlier pada setiap fitur (kecuali target) ditangani menggunakan metode IQR (Interquartile Range) capping, di mana nilai di luar rentang [Q1 − 1.5 × IQR,Q3 + 1.5 × IQR] diganti dengan batas bawah atau batas atas rentang tersebut.
- Feature selection: Dataset dibagi menjadi dua bagian fitur (variabel independen X) seluruh kolom kecuali PCOS_Diagnosis, dan target (variabel independen y) yaitu kolom PCOS_Diagnosis.
- Data Normalization/Scaling: Normalisasi fitur numerik seperti Age, BMI, Testosterone_Level, dan Antral_Follicle_Count agar Logistic Regression dapat bekerja optimal.
```
# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
- Data Splitting: Train-test split dengan rasio 80:20.
```
# Split data (training 80% dan testing 20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```
- Penanganan ketidakseimbangan target
```
# Penanganan data tidak seimbang
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

## Modeling
1. Logistic Regression
-- Model Logistic Regression adalah model linear classification yang digunakan untuk memprediksi probabilitas kelas (0 = tidak terindikasi PCOS, 1 = terindikasi PCOS)
-- Model ini diinisialisasi dengan random_state=42 dan solver='liblinear' (cocok untuk dataset kecil) dan dilatih menggunakan X_train_resampled dan y_train_resampled.

3. Random Forest Classifier
-- Model Random Forest adalah ensemble method berbasu decision tree yang menggabubgkan banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting.
-- Model ini diinisialisasi dengan n_estimators=100 dan random_state=42 sebagai baseline dan dilatih menggunakan X_train_resampled dan y_train_resampled.

5. Hyperparameter Tuning (Random Forest):
-- Menggunakan GridSearchCV untuk parameter:
```
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 4, 6],
    'max_features': ['sqrt', 'log2']
}
```
-- Tujuannya untuk meningkatkan performa dan menghindari overfitting. Tetapi setelah diterapkan model tetap overfitting, kemungkinan karena dataset sintesis.

## Evaluation
### Metrik Evaluasi:
1. Akurasi:  Proporsi prediksi yang benar (TP + TN) dibagi dengan total jumlah prediksi.
2. Precision, Recall, F1-Score: Untuk mengukur performa pada data tidak seimbang.
3. Mean Squared Error (MSE): Rata-rata kuadrat perbedaan antara nilai aktual dan prediksi
4. Confusion Matrix: Visualisasi hasil prediksi.
5. Classification Report: Ringkasan Precision, Recall, dan F1-Score per kelas, serta Accuracy.
(TP: True Positive, TN: True Negative, FP: False Positive, FN: False Negative)

### Hasil Evaluasi:
| Model                    | Accuracy | Precision | Recall  | F1 Score | MSE    |
| ------------------------ |:--------:| ---------:| ------- |:--------:|-------:|
| Logistic Regression      | 0.9983   |  1.0      | 0.9928  | 0.9964   | 0.0017 |
| Random Forest (Baseline) | 1.0000   |  1.0      | 1.0000  | 1.0000   | 0.0000 |
| Random Forest (Tuned)    | 1.0000   |  1.0      | 1.0000  | 1.0000   | 0.0000 |

### Visualisasi:
1. Logistic Regression

![Screenshot (1521)](https://github.com/user-attachments/assets/8652249c-548c-44e5-b78f-3c33ac9f875a)

2. Random Forest
![Screenshot (1522)](https://github.com/user-attachments/assets/2178e977-29fb-40de-b2e3-5fb1cea3594b)
![Screenshot (1523)](https://github.com/user-attachments/assets/e55bc322-5566-4d7f-b614-0f012d2f15da)


