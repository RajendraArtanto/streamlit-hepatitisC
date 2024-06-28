import pickle
import numpy as np
import streamlit as st

# Fungsi untuk memuat model
def load_model(model_name):
    return pickle.load(open(model_name, 'rb'))

# Model names
model_names = {
    'Logistic Regression': 'logistic_regression_hepatitis_c.sav',
    'Random Forest': 'random_forest_hepatitis_c.sav',
    'K-Nearest Neighbors': 'knn_hepatitis_c.sav',
    'Decision Tree': 'decision_tree_hepatitis_c.sav',
    'CatBoost': 'catboost_hepatitis_c.sav',
    'Gradient Boosting': 'gradient_boosting_hepatitis_c.sav'
}

# Judul web
st.title('Prediksi Penyakit Hepatitis C')

# Memilih model
selected_model = st.selectbox('Pilih Model Prediksi', list(model_names.keys()))
model_path = model_names[selected_model]
model = load_model(model_path)

# Mengatur input kolom
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input('Umur', min_value=0, max_value=120, value=0)

with col2:
    Sex = st.number_input('Jenis Kelamin (1 = Pria, 2 = Wanita)', min_value=1, max_value=2, value=1)

with col3:
    ALB = st.number_input('Jumlah Kadar Alburnin (ALB)', min_value=0.0, step=0.01, value=0.0)

with col1:
    ALP = st.number_input('Jumlah Kadar Alkaline Phosphatase (ALP)', min_value=0.0, step=0.01, value=0.0)

with col2:
    ALT = st.number_input('Jumlah Kadar Alanin Transaminase (ALT)', min_value=0.0, step=0.01, value=0.0)

with col3:
    AST = st.number_input('Jumlah Kadar Aspartat (AST)', min_value=0.0, step=0.01, value=0.0)

with col1:
    BIL = st.number_input('Jumlah Kadar Bilirubin (BIL)', min_value=0.0, step=0.01, value=0.0)

with col2:
    CHE = st.number_input('Jumlah Kadar Kolinesterase (CHE)', min_value=0.0, step=0.01, value=0.0)

with col3:
    CHOL = st.number_input('Jumlah Kadar Kolesterol (CHOL)', min_value=0.0, step=0.01, value=0.0)

with col1:
    CREA = st.number_input('Jumlah Kadar Kreatin (CREA)', min_value=0.0, step=0.01, value=0.0)

with col2:
    GGT = st.number_input('Jumlah Kadar Gamma-Glutamil-Transferase (GGT)', min_value=0.0, step=0.01, value=0.0)

with col3:
    PROT = st.number_input('Jumlah Kadar Protein (PROT)', min_value=0.0, step=0.01, value=0.0)

# Kode prediksi
hepatitis_c_diagnosis = ''
probability = ''

# Tombol prediksi
if st.button('Prediksi Penyakit Hepatitis C'):
    Sex = 0 if Sex == 1 else 1  # Konversi nilai 1 menjadi 0 untuk pria dan 2 menjadi 1 untuk wanita
    input_data = np.array([[Age, Sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT]])
    hepatitis_c_diagnosis = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    if hepatitis_c_diagnosis[0] == 1:
        hepatitis_c_diagnosis = 'Pasien Terkena penyakit Hepatitis C'
        probability = f"Probabilitas: {probabilities[0][1] * 100:.2f}%"
    else:
        hepatitis_c_diagnosis = 'Pasien Tidak Terkena penyakit Hepatitis C'
        probability = f"Probabilitas: {probabilities[0][0] * 100:.2f}%"

    st.success(hepatitis_c_diagnosis)
    st.info(probability)
