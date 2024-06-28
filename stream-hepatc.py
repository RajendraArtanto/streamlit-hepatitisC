import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Fungsi untuk memuat model
def load_model(model_name):
    return pickle.load(open(model_name, 'rb'))

# Data untuk tabel
data = {
    'Atribut': ['Umur', 
                'Jenis Kelamin', 
                'Jumlah Kadar ALB', 
                'Jumlah Kadar ALP',
                'Jumlah Kadar ALT',
                'Jumlah Kadar AST',
                'Jumlah Kadar BIL',
                'Jumlah Kadar CHE',
                'Jumlah Kadar CHOL',
                'Jumlah Kadar CREA',
                'Jumlah Kadar GGT',
                'Jumlah Kadar PROT',
                ],
    'Deskripsi': ['32 - 77', 
                  '1 = Pria, 2 = Wanita', 
                  '14,9 - 82,2', 
                  '11,3 - 416,6',
                  '0,9 - 325,3',
                  '10,6 - 324',
                  '0,8 - 209',
                  '1,42 - 16,41',
                  '1,43 - 9,67',
                  '8 - 1079,1',
                  '4,5 - 650,9',
                  '44,8 - 86,5',
                  ]
}

# Membuat DataFrame
df = pd.DataFrame(data)

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
st.title('Aplikasi Prediksi Penyakit Hepatitis C')

# Nama
st.write("Oleh : Rajendra Artanto - 21.11.4236")
st.write("Aplikasi ini bertujuan untuk membantu tenaga medis dalam mendiagnosis dan memprediksi penyakit Hepatitis C. Dengan menggunakan teknologi kecerdasan buatan dan analisis data, aplikasi ini mampu memberikan prediksi yang akurat berdasarkan data medis pasien, seperti hasil tes laboratorium, riwayat kesehatan, dan faktor risiko lainnya. Dengan aplikasi ini, tenaga medis dapat meningkatkan akurasi diagnosis, mempercepat proses pengambilan keputusan, dan memberikan perawatan yang lebih efektif kepada pasien yang berisiko atau sudah terdiagnosis Hepatitis C.")

# Menampilkan tabel
st.write('Tabel Petunjuk Pengisian')
st.dataframe(df)

# Fungsi untuk menampilkan deskripsi dalam kolom-kolom
def show_descriptions():
    # Membagi layar menjadi tiga kolom
    col1, col2, col3 = st.columns(3)

    # Deskripsi ALB
    with col1:
        st.info("Albumin (ALB) adalah protein yang diproduksi oleh hati dan merupakan komponen penting dalam menjaga tekanan osmotik darah dan transportasi zat-zat seperti hormon dan obat-obatan.")

    # Deskripsi ALP
    with col2:
        st.warning("Alkaline Phosphatase (ALP) adalah enzim yang terdapat di hati, tulang, usus, dan ginjal. Peningkatan kadar ALP dapat menunjukkan masalah pada hati atau tulang.")

    # Deskripsi ALT
    with col3:
        st.success("Alanine Aminotransferase (ALT) adalah enzim yang hadir terutama di hati. Kadar ALT yang tinggi dapat menandakan adanya kerusakan pada sel-sel hati.")

    # Deskripsi AST
    with col1:
        st.info("Aspartate Aminotransferase (AST) adalah enzim yang juga ditemukan di hati, otot jantung, otot rangka, dan ginjal. Peningkatan AST dapat menunjukkan adanya kerusakan pada jaringan-jaringan ini.")

    # Deskripsi BIL
    with col2:
        st.warning("Bilirubin (BIL) adalah pigmen kuning yang dihasilkan dari pemecahan hemoglobin dalam sel darah merah. Kadar bilirubin yang tinggi dapat menunjukkan masalah pada hati atau gangguan dalam pemecahan sel darah merah.")

    # Deskripsi CHE
    with col3:
        st.success("Cholinesterase (CHE) adalah enzim yang bertanggung jawab untuk mengatur kolin dalam tubuh. Kadar CHE dapat memberikan informasi tentang fungsi hati dan keracunan organofosfat.")

    # Deskripsi CHOL
    with col1:
        st.info("Cholesterol (CHOL) adalah senyawa lemak yang ditemukan di dalam sel-sel tubuh dan dalam darah. Kadar kolesterol yang tinggi dapat meningkatkan risiko penyakit jantung.")

    # Deskripsi CREA
    with col2:
        st.warning("Creatinine (CREA) adalah hasil samping metabolisme otot yang diekskresikan oleh ginjal. Kadar creatinine dalam darah dapat memberikan informasi tentang fungsi ginjal.")

    # Deskripsi GGT
    with col3:
        st.success("Gamma-Glutamyl Transferase (GGT) adalah enzim yang terdapat di hati, pankreas, ginjal, dan otot jantung. Peningkatan GGT dapat menunjukkan adanya kerusakan hati atau gangguan dalam aliran empedu.")

    # Deskripsi PROT
    with col1:
        st.info("Total Protein (PROT) mengukur jumlah total protein dalam darah, yang terdiri dari albumin dan globulin. Kadar protein total dapat memberikan informasi tentang status gizi, kondisi ginjal, dan hati.")

# Judul button dropdown
st.subheader('Penjelasan Lebih Lanjut')
# Membuat button dropdown
with st.expander("Klik di sini untuk melihat penjelasan lebih lanjut"):
    show_descriptions()

# Memilih model
st.subheader('Input Data')
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
    ALB = st.number_input('Jumlah Kadar Alburnin (ALB)', min_value=0.0, step=0.1, value=0.0)

with col1:
    ALP = st.number_input('Jumlah Kadar Alkaline Phosphatase (ALP)', min_value=0.0, step=0.1, value=0.0)

with col2:
    ALT = st.number_input('Jumlah Kadar Alanin Transaminase (ALT)', min_value=0.0, step=0.1, value=0.0)

with col3:
    AST = st.number_input('Jumlah Kadar Aspartat (AST)', min_value=0.0, step=0.1, value=0.0)

with col1:
    BIL = st.number_input('Jumlah Kadar Bilirubin (BIL)', min_value=0.0, step=0.1, value=0.0)

with col2:
    CHE = st.number_input('Jumlah Kadar Kolinesterase (CHE)', min_value=0.0, step=0.1, value=0.0)

with col3:
    CHOL = st.number_input('Jumlah Kadar Kolesterol (CHOL)', min_value=0.0, step=0.1, value=0.0)

with col1:
    CREA = st.number_input('Jumlah Kadar Kreatin (CREA)', min_value=0.0, step=0.1, value=0.0)

with col2:
    GGT = st.number_input('Jumlah Kadar Gamma-Glutamil-Transferase (GGT)', min_value=0.0, step=0.1, value=0.0)

with col3:
    PROT = st.number_input('Jumlah Kadar Protein (PROT)', min_value=0.0, step=0.1, value=0.0)

# Kode prediksi
hepatitis_c_diagnosis = ''
probability = ''

# Tombol prediksi
if st.button('Prediksi'):
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
