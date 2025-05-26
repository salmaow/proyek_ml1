# Laporan Proyek Machine Learning - Salma Oktarina
## Domain Proyek
### Latar Belakang:
PCOS atau sindrom polikistik ovarium merupakan gangguan hormonal yang ditandai dengan adanya kista atau kantung kecil berisi cairan pada ovarium. Penyebab pasti PCOS belum diketahui, namun sejumlah faktor yang berkontribusi antara lain resistensi insulin, kelebihan hormon androgen, dan siklus menstruasi yang tidak teratur.

PCOS dapat menyebabkan terganggunya ovulasi dan produksi hormon estrogen, yang berujung pada gangguan kesuburan. Deteksi dini sangat penting untuk mencegah komplikasi seperti infertilitas, diabetes tipe 2, dan masalah jantung.

### Masalah:
Banyak wanita yang tidak menyadari bahwa mereka memiliki PCOS karena gejalanya seringkali tidak disadari atau dianggap hal biasa. Maka dari itu, perlu adanya sistem prediksi yang dapat mengklasifikasikan kemungkinan PCOS berdasarkan gejala dan indikator medis tertentu.

### Solusi:
Dengan memanfaatkan machine learning, kita dapat membangun model klasifikasi untuk memprediksi kemungkinan seseorang mengalami PCOS berdasarkan data medis dan gejala yang dialami.

## Business Understanding
### Problem Statements
- Bagaimana memanfaatkan data medis untuk mengklasifikasikan pasien yang kemungkinan mengalami PCOS?
- Algoritma machine learning apa yang paling efektif untuk memprediksi kondisi PCOS berdasarkan data yang tersedia?

### Goals
- Mengembangkan model klasifikasi untuk mendeteksi kemungkinan PCOS dari data medis.
- Membandingkan performa model Logistic Regression dan Random Forest.

### Solution Statements
- Menggunakan algoritma Logistic Regression sebagai baseline karena cepat dan mudah diinterpretasikan.
- Menggunakan Random Forest untuk mengatasi fitur non-linear dan meningkatkan akurasi.
- Melakukan hyperparameter tuning pada Random Forest jika diperlukan untuk meningkatkan performa model.

## Data Understanding
Dataset yang digunakan diambil dari Kaggle - Polycystic Ovary Syndrome (PCOS). Dataset ini berisi data demografi dan hasil pemeriksaan medis. 

Dalam proyek ini, fituryang digunakan sebagai berikut:
- Age: Umur
- BMI: Indeks massa tubuh
- Menstrual_Irregularity: Ketidakteraturan siklus menstruasi (0/1)
- Testosterone_Level: Kadar hormon testosteron (ng/dL)
- Antral_Follicle_Count: Jumlah folikel antral berdasarkan USG
- PCOS_Diagnosis: Label target (1 jika positif PCOS, 0 jika tidak)

## Data Preparation
Langkah-langkah data preparation:
- Handling missing values: Menghapus baris dengan nilai kosong jika ada.
- Encoding: Mengubah nilai kategorikal Menstrual_Irregularity (Y/N) menjadi numerik (1/0).
- Feature selection: Hanya menggunakan lima fitur utama.
- Scaling: Normalisasi fitur numerik untuk model Logistic Regression.
- Split dataset: Train-test split dengan rasio 80:20.

## Modeling
- Model 1: Logistic Regression
Model baseline untuk klasifikasi biner.
Cocok untuk interpretasi dan perbandingan awal.

- Model 2: Random Forest Classifier
Cocok untuk menangani interaksi non-linear antar fitur.
Parameter awal: n_estimators=100, max_depth=None, random_state=42.

- Hyperparameter Tuning (Jika akurasi awal rendah):
Menggunakan GridSearchCV untuk parameter:
n_estimators: 100, 200, 300
max_depth: 5, 10, 15
max_features: 'sqrt', 'log2'
min_samples_split: 2, 4, 6

## Evaluation
### Metrik Evaluasi:
Akurasi: Proporsi prediksi benar terhadap total data.
Precision, Recall, F1-Score: Untuk mengukur performa pada data tidak seimbang.
Confusion Matrix: Visualisasi hasil prediksi.

### Hasil Evaluasi:
Contoh hasil (ilustratif):
Logistic Regression: Akurasi = 84%, F1-score = 0.81
Random Forest: Akurasi = 89%, F1-score = 0.87

### Visualisasi:
Confusion matrix menggunakan seaborn heatmap.
Grafik feature importance untuk Random Forest.

## Referensi
[1] Lucas Santos, "Polycystic Ovary Syndrome (PCOS) Dataset", Kaggle, 2022. [Online]. Available: https://www.kaggle.com/datasets/lucass0s0/polycystic-ovary-syndrome-pcos

[2] P. Patel, et al., "A Comprehensive Approach to Predicting Polycystic Ovary Syndrome Using Machine Learning", IJCRT, 2024. [Online]. Available: https://ijcrt.org/papers/IJCRT24A5401.pdf

[3] S. Raschka and V. Mirjalili, Python Machine Learning, 3rd ed., Packt Publishing, 2019.
