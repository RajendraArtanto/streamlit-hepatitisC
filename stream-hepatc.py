import pickle
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

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

# Sidebar menu
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih halaman:", ["Home", "Prediksi", "Visualisasi"])

if menu == "Home":
    # Judul web
    st.title('Sistem Prediksi Penyakit Hepatitis-C')
    st.write("Oleh: Rajendra Artanto - 21.11.4236")

    # Menambahkan gambar di bawah judul
    st.image('asset/heading.jpg', use_column_width=True)

    st.write("Aplikasi ini bertujuan untuk membantu tenaga medis dalam mendiagnosis dan memprediksi penyakit Hepatitis C. Dengan menggunakan teknologi kecerdasan buatan dan analisis data, aplikasi ini mampu memberikan prediksi yang akurat berdasarkan data medis pasien, seperti hasil tes laboratorium, riwayat kesehatan, dan faktor risiko lainnya. Dengan aplikasi ini, tenaga medis dapat meningkatkan akurasi diagnosis, mempercepat proses pengambilan keputusan, dan memberikan perawatan yang lebih efektif kepada pasien yang berisiko atau sudah terdiagnosis Hepatitis C.")
    
    # Menampilkan tabel
    st.subheader('Tabel Petunjuk Pengisian')
    st.dataframe(df)
    
    # Judul button dropdown
    st.subheader('Penjelasan Atribut')
    
    # Membuat button dropdown
    with st.expander("Klik di sini untuk melihat penjelasan lebih lanjut"):
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

    st.subheader("Model - Model Prediksi")
    with st.expander("Logistic Regression"):
        st.image('asset/logisticregression.jpg',use_column_width=True)
        st.write("Logistic Regression adalah salah satu algoritma dalam machine learning yang digunakan untuk klasifikasi data biner. Algoritma ini menggunakan pendekatan regresi linier untuk memodelkan hubungan antara variabel input dan output, dengan menghasilkan nilai probabilitas yang berkisar antara 0 dan 1. Dalam hal ini, output klasifikasi dilakukan berdasarkan probabilitas tersebut.")
        st.write("Logistic Regression menghitung nilai probabilitas suatu instance data masuk ke dalam kelas tertentu. Hal ini dilakukan dengan memperhitungkan bobot setiap variabel input pada suatu fungsi logistik. Fungsi logistik digunakan untuk mengubah hasil perhitungan bobot variabel input menjadi nilai probabilitas yang berkisar antara 0 dan 1.")
        st.write("Dalam proses training, Logistic Regression meminimalkan error prediksi dengan memperbarui bobot variabel input menggunakan teknik optimasi seperti Gradient Descent. Tujuannya adalah untuk menemukan bobot yang terbaik sehingga model dapat memprediksi kelas dari instance data yang belum dilihat dengan akurasi yang tinggi.")

    with st.expander("Decision Tree"):
        st.image('asset/decisiontree.png', use_column_width=True)
        st.write("Decision Tree adalah algoritma flowchart berbentuk struktur pohon yang digunakan untuk membantu membuat keputusan atau menyelesaikan tugas yang berkaitan dengan regresi dan klasifikasi. Struktur decision tree dimulai dari simpul akar (root node), cabang, simpul internal (internal node/decision node), dan terakhir simpul daun (leaf node/terminal node). ")
        st.write("Simpul akar (root node) mewakili pertanyaan atau masalah yang ingin dipecahkan. Kemudian cabang merupakan jalur keputusan, yang nantinya akan mengarah ke beberapa keputusan atau internal node. Setiap decision tree bisa memiliki beberapa internal node sebagai alternatif jawaban atau keputusan. Internal node juga bisa memiliki cabang node lain yaitu leaf node, yang akan mewakili keputusan akhir. ")

    with st.expander("Random Forest"):
        st.image('asset/srandomforest.png', use_column_width=True)
        st.write("Random forest adalah algoritma yang menggabungkan hasil (output) dari beberapa decision tree untuk mencapai satu hasil yang lebih akurat. Random forest membutuhkan gabungan beberapa decision tree untuk memprediksi hasil yang akurat. ")
        st.write("Konsep sederhana dari random forest adalah beberapa decision tree yang tidak berkorelasi akan bekerja lebih baik sebagai kelompok dibandingkan individu. Saat menggunakan random forest sebagai pengklasifikasi, satu decision tree menyumbang satu suara. Setiap decision tree bisa menghasilkan jawaban yang sama atau berbeda satu sama lain. ")
        st.write("Misalnya decision tree A, B, E dan F memprediksi hasil 1. Sementara decision tree C dan D memprediksi hasil 0. Karena ada banyaknya alternatif jawaban dalam decision tree dan kemungkinan bias yang tinggi, random forest mengambil prediksi hasil dari beberapa decision tree berdasarkan suara mayoritas dan memprediksi hasil yang lebih akurat.")
        st.write("Semakin banyak hasil decision tree yang diambil, semakin tinggi akurasi terutama ketika masing-masing pohon tidak berkorelasi satu sama lain.  ")

    with st.expander("K-Nearest Neighbors"):
        st.image('asset/knn.jpg',use_column_width=True)
        st.write("Nearest Neighbor atau k-Nearest Neighbor (kNN) merupakan salah satu algoritme klasifikasi dalam data mining yang memanfaatkan data terdekat untuk melakukan prediksi pada data baru yang belum dikenal (data uji). Algoritme ini bekerja dengan cara mencari sejumlah tetangga terdekat dari data uji dan menentukan kelas data uji tersebut berdasarkan mayoritas kelas dari tetangga terdekat (data latih) yang ditemukan.")
        st.write("Nearest Neighbor dapat digunakan untuk menangani berbagai jenis data, baik data numerik maupun kategorikal. Pada data kategorikal, perhitungan jarak perbedaan atau kesamaan tidak dapat dihitung menggunakan operasi matematik seperti yang dapat dilakukan pada data numerik. Nearest Neighbor lebih efektif pada data dengan dimensi yang rendah atau sedang.")
        st.write("Selain itu, algoritma ini juga efektif untuk dataset dengan jumlah data yang kecil hingga sedang, karena semakin besar jumlah data yang digunakan maka waktu yang dibutuhkan untuk melakukan klasifikasi semakin lama. Tak hanya memiliki kelebihan, kNN juga memiliki kekurangan seperti sensitif terhadap nilai pencilan (outlier) dan ketidakseimbangan kelas (class imbalance).")

    with st.expander("CatBoost"):
        st.image('asset/catboost.jpg',use_column_width=True)
        st.write("CatBoost adalah algoritma Boosting open-source yang dikembangkan oleh tim Yandex, perusahaan teknologi terkemuka di Rusia. CatBoost merupakan singkatan dari “Category Boosting,” yang menunjukkan keunggulan algoritma ini dalam menangani fitur kategorikal dalam data. Algoritma CatBoost dikembangkan untuk meningkatkan performa model Machine Learning dengan fokus pada kecepatan, akurasi, dan kemampuan penanganan fitur kategorikal.")
        #st.write("CatBoost didasarkan pada algoritma Gradient Boosting yang telah terbukti sukses, namun dengan adanya inovasi dan penyesuaian tertentu. Pengembang CatBoost memperkenalkan beberapa fitur dan teknik unik yang membedakannya dari algoritma Boosting lainnya, termasuk penanganan otomatis terhadap fitur kategorikal, penanganan missing values, dan fitur “Ordered Boosting” yang membantu mengatasi overfitting.")
        st.write("CatBoost memiliki beberapa keunggulan utama, termasuk kemampuannya dalam menangani data yang tidak terstruktur dan fitur kategorikal tanpa memerlukan encoding manual, yang mengurangi kompleksitas pra-pemrosesan data. Algoritma ini juga secara otomatis menangani missing values dengan memperkirakan nilai yang hilang berdasarkan informasi dari fitur lainnya dalam data. ")
        st.write("Selain itu, CatBoost memiliki fitur “Ordered Boosting” yang membantu mengatasi overfitting dengan menyesuaikan batas-batas kompleksitas model pada setiap iterasi, sehingga menghasilkan model yang lebih umum dan akurat. Terakhir, CatBoost dirancang untuk memberikan kinerja tinggi dan kecepatan training yang efisien, memungkinkan pelatihan model dengan cepat dan efektif, baik pada dataset kecil maupun besar.")
    
    with st.expander("Gradient Boosting"):
        st.image('asset/gradientboosting.png',use_column_width=True)
        st.write("Gradient Boosting adalah salah satu metode Machine Learning yang berfokus pada perbaikan kinerja model melalui peningkatan performa model sebelumnya. Algoritma ini menggunakan pendekatan boosting yang melibatkan peningkatan performa model dengan memanfaatkan informasi dari model-model sebelumnya.")
        st.write("Gradient Boosting merupakan algoritma machine learning yang menggabungkan beberapa model kecil menjadi satu model yang lebih kuat dan lebih baik dalam memprediksi data. Algoritma ini bekerja dengan mengukur eror dari model sebelumnya dan menggunakan informasi tersebut untuk memperbaiki performa model berikutnya.")
        st.write("Gradient Boosting terkenal akan kemampuannya untuk menangani data yang kompleks dan memberikan prediksi yang akurat dengan menggabungkan kekuatan dari beberapa model lemah untuk membentuk model yang kuat. Teknik ini juga fleksibel dan dapat digunakan untuk berbagai jenis data, baik regresi maupun klasifikasi.")
        st.write("Namun, kelemahan Gradient Boosting sendiri adalah bahwa proses trainingnya cenderung lambat dan membutuhkan sumber daya komputasi yang tinggi, terutama pada dataset besar. Selain itu, model ini rentan terhadap overfitting jika tidak dikonfigurasi dengan baik, memerlukan pemilihan parameter yang hati-hati dan validasi yang tepat untuk mencapai kinerja optimal.")

elif menu == "Prediksi":
    st.subheader('Input Data')
    selected_model = st.selectbox('Pilih Model Prediksi', list(model_names.keys()))
    model_path = model_names[selected_model]
    model = load_model(model_path)

    st.info("Pengisian atribut dapat menggunakan contoh rentang yang terdapat pada tabel petunjuk pengisian")

    # Mengatur input kolom
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input('Umur', min_value=0, max_value=120, value=0)

    with col2:
        Sex = st.number_input('Jenis Kelamin (1 = Pria, 2 = Wanita)', min_value=1, max_value=2, value=1)

    with col3:
        ALB = st.number_input('Jumlah Kadar Albumin (ALB)', min_value=0.0, step=0.1, value=0.0)

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

        # Visualisasi hasil prediksi
        st.subheader('Visualisasi Hasil Prediksi')
        probabilities = probabilities[0]
        labels = ['Negatif', 'Positif']
        plt.bar(labels, probabilities, color=['blue', 'red'])
        plt.xlabel('Prediksi')
        plt.ylabel('Probabilitas')
        st.pyplot(plt)
        
elif menu == "Visualisasi":
    st.subheader('Visualisasi Dataset Hepatitis C')

    st.write("Dataset yang digunakan yaitu dataset pasien Hepatitis C yang diambil dari situs Kaggle.com dan telah dilakukan proses EDA. EDA, atau Exploratory Data Analysis, adalah proses analisis awal data yang bertujuan untuk memahami karakteristik, struktur, dan komponen penting dari dataset sebelum melakukan analisis statistik atau pemodelan prediktif lebih lanjut.")
    st.write("Klik link dibawah ini untuk mengakses dataset")
    st.write("[Dataset asli](https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset) | [Dataset + EDA](https://drive.google.com/file/d/1dErBu_QUdeIBK9bQNUYa6zZvGZ2-_Yat/view?usp=drive_link)")
    
    # Baca dataset dari file CSV
    df_visual = pd.read_csv('HepatitisCdata_modified.csv')

    # Pilihan plot
    plot_type = st.selectbox('Pilih jenis plot', ['Scatter Plot', 'Histogram', 'Box Plot'])

    if plot_type == 'Scatter Plot':
        x_axis = st.selectbox('Pilih X-axis', df_visual.columns[:-1])
        y_axis = st.selectbox('Pilih Y-axis', df_visual.columns[:-1])
        fig, ax = plt.subplots()
        sns.scatterplot(x=x_axis, y=y_axis, hue='Category', data=df_visual, ax=ax)
        st.pyplot(fig)

    elif plot_type == 'Histogram':
        column = st.selectbox('Pilih kolom untuk histogram', df_visual.columns[:-1])
        fig, ax = plt.subplots()
        sns.histplot(df_visual[column], kde=True, ax=ax)
        st.pyplot(fig)

    elif plot_type == 'Box Plot':
        column = st.selectbox('Pilih kolom untuk box plot', df_visual.columns[:-1])
        fig, ax = plt.subplots()
        sns.boxplot(x='Category', y=column, data=df_visual, ax=ax)
        st.pyplot(fig)
