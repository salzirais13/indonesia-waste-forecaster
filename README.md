# Indonesia Waste Prediction Web App

Repository ini berisi pipeline end-to-end untuk pembersihan data, imputasi nilai hilang, pemodelan berbasis Recurrent Neural Network (RNN), hingga deployment model prediksi timbunan sampah (TS) tingkat provinsi di Indonesia menggunakan Flask.

## 📌 Fitur Utama
* **Data Imputation & Preprocessing (R):** Menangani *missing values* pada data runtun waktu timbunan sampah kab/kota menggunakan algoritma KNN Imputer (k=6) dan Interpolasi Linear.
* **Aggregasi Provinsi:** Mengelompokkan total sampah tahunan per provinsi (`hotdeckcsv.csv`) sebagai basis training model nasional.
* **Deep Learning Forecasting (Python & TensorFlow):** Implementasi arsitektur `SimpleRNN` untuk memproyeksikan tren sampah ke masa depan dengan mekanisme *look-back* 3 tahun.
* **Dashboard Interaktif (Flask):** UI berbasis web untuk memprediksi volume sampah tahunan dan persentase kejenuhan kapasitas TPA harian berdasarkan provinsi dan tahun target.

---

## 📂 Struktur Repositori

* `data/`: Berisi dataset mentah (`raw`) dan dataset hasil olahan (`processed`).
  * `hotdeckcsv.csv` merupakan data hasil komputasi untuk menangani missing value menggunakan hotdeck.
* `scripts/`: Script pemrograman R untuk eksperimen imputasi awal dan prototyping model.
* `web_app/`: Source code aplikasi web Flask, UI template, aset visual, dan konfigurasi deployment.

---

## 🛠️ Cara Menjalankan Aplikasi Web

### 1. Prasyarat
Pastikan Anda sudah menginstal Python (versi 3.9 - 3.11 direkomendasikan).

### 2. Instalasi Dependency
Masuk ke folder aplikasi web dan instal package yang dibutuhkan:
```bash
cd web_app
pip install -r requirements.txt
```

### 3. Jalankan Server Flask
Eksekusi perintah berikut untuk memulai server lokal:
```bash
python app.py
```
Buka browser Anda dan akses halaman `http://127.0.0.1:5000/`.

---

## 📊 Metodologi Prediksi Kapasitas TPA
Persentase beban kapasitas Tempat Pemrosesan Akhir (TPA) per hari dihitung menggunakan rumus berikut:

Saturasi Landfill (%) = ((Estimasi Sampah Tahunan / 365) / Kapasitas TPA Maksimal) * 100
