import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(file_path):
    """Memuat data dari file CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di {file_path}")
    return pd.read_csv(file_path)

def build_preprocessor():
    """Membangun Pipeline preprocessing dengan ColumnTransformer."""
    # Definisikan kolom berdasarkan tipe
    numeric_features = ['Age', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    
    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk fitur kategorikal
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Gabungkan dengan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Buang kolom seperti Name, Ticket, Cabin, PassengerId yang butuh advanced feature engineering atau tidak relevan
    )
    
    return preprocessor

def run_preprocessing(input_path, output_dir):
    """Menjalankan proses keseluruhan dan menyimpan hasilnya."""
    print("[INFO] Memulai preprocessing data...")
    
    # Load data
    df = load_data(input_path)
    
    # Pisahkan target dan fitur
    if 'Survived' not in df.columns:
        raise ValueError("Kolom target 'Survived' tidak ditemukan!")
        
    y = df['Survived']
    X = df.drop('Survived', axis=1)
    
    # Build dan fit_transform preprocessor
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Dapatkan nama fitur yang baru
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_features = cat_encoder.get_feature_names_out(categorical_features)
    numeric_features = ['Age', 'Fare']
    all_features = numeric_features + list(cat_features)
    
    # Buat DataFrame baru
    X_processed_df = pd.DataFrame(X_processed, columns=all_features)
    
    # Gabungkan kembali untuk disimpan sebagai satu file CSV yang bersih (jika diperlukan untuk kriteria)
    df_preprocessed = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)
    
    # Pastikan direktori output ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan dataset hasil preproses
    output_csv = os.path.join(output_dir, 'titanic_preprocessed.csv')
    df_preprocessed.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Data hasil preprocessing disimpan di {output_csv}")
    
    # Simpan preprocessor (opsional tapi disarankan untuk inference)
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[SUCCESS] Preprocessor object disimpan di {preprocessor_path}")
    
    return X_processed_df, y

if __name__ == "__main__":
    # Konfigurasi path relatif terhadap letak script ini dijalankan
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.join(BASE_DIR, '..', 'titanic_raw', 'titanic.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'titanic_preprocessing')
    
    # Eksekusi
    X_train, y_train = run_preprocessing(RAW_DATA_PATH, OUTPUT_DIR)
    
    print("\n[INFO] Sample data X (5 baris pertama):")
    print(X_train.head())
    print("\n[INFO] Preprocessing Selesai.")
