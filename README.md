# Eksperimen SML - Mochamad Chairulridjal Nurvikri

Repository ini berisi eksperimen terhadap dataset pelatihan untuk memenuhi Kriteria 1 (Advanced) dari submission akhir Dicoding: Membangun Sistem Machine Learning.

## Struktur Direktori
```
Eksperimen_SML_MochamadChairulridjalNurvikri
├── .github
│   └── workflows
│       └── preprocess.yml
├── titanic_raw
│   └── titanic.csv
├── preprocessing
│   ├── Eksperimen_MochamadChairulridjalNurvikri.ipynb
│   ├── automate_MochamadChairulridjalNurvikri.py
│   └── titanic_preprocessing
│       ├── preprocessor.pkl
│       └── titanic_preprocessed.csv
└── README.md
```

## Penjelasan

1. **Eksperimen_MochamadChairulridjalNurvikri.ipynb**: Jupyter notebook berisi proses Data Loading, Exploratory Data Analysis (EDA) dengan 10 visualisasi, dan Data Preprocessing secara manual.
2. **automate_MochamadChairulridjalNurvikri.py**: Script Python yang mengotomatisasi proses preprocessing menggunakan `sklearn.pipeline.Pipeline` dan `ColumnTransformer`. Script ini akan menghasilkan file `titanic_preprocessed.csv` dan model `preprocessor.pkl`.
3. **.github/workflows/preprocess.yml**: Workflow GitHub Actions yang mendeteksi perubahan pada `titanic_raw/` atau file script, lalu otomatis menjalankan preprocessing dan push hasil perubahannya (caching pip diaktifkan).

## Cara Menjalankan Preprocessing Secara Lokal

1. Pastikan Anda memiliki Python 3.12 terinstal.
2. Install dependency:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```
3. Buka terminal di direktori `preprocessing/` dan jalankan:
   ```bash
   python automate_MochamadChairulridjalNurvikri.py
   ```
4. Anda akan melihat log berhasil dan file CSV baru di dalam folder `titanic_preprocessing`.

## Menjalankan Notebook
Gunakan ekstensi Jupyter di VSCode atau jalankan `jupyter notebook` lalu buka file `Eksperimen_MochamadChairulridjalNurvikri.ipynb` untuk melihat visualisasi dan langkah-langkah preprocessing manual.
