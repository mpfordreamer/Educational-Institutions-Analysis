# **Proyek Data Science: Meningkatkan Retensi dan Keberhasilan Akademik di Jaya Jaya Institute Education**

## **Business Understanding**
![jayajaya](https://github.com/user-attachments/assets/9205e4a5-4db9-4f16-8a0a-b509f177d367)

**Jaya Jaya Institute Education** saat ini menghadapi tantangan signifikan terkait tingkat putus sekolah (*dropout*), status mahasiswa aktif (*enrolled*), dan keberhasilan kelulusan (*graduate*). Angka *dropout* yang tinggi tidak hanya merugikan reputasi institusi, tetapi juga menyia-nyiakan sumber daya yang telah diinvestasikan dalam pendidikan mahasiswa. Dengan pemahaman mendalam tentang faktor-faktor yang memengaruhi keputusan mahasiswa untuk melanjutkan atau menghentikan studi, institusi dapat merancang dan mengimplementasikan strategi intervensi yang lebih cerdas dan efektif untuk meningkatkan retensi serta memastikan capaian akademik yang lebih tinggi.

### **Permasalahan Bisnis Esensial:**

Terlepas dari beragam program studi yang ditawarkan oleh Jaya Jaya Institute Education, terdapat variasi yang mencolok dalam tingkat retensi dan performa akademik di setiap jurusan. Tingginya angka *dropout* dan rendahnya kinerja akademik di beberapa program studi mengindikasikan adanya faktor-faktor pendorong yang belum sepenuhnya dipahami atau ditangani secara optimal. Oleh karena itu, diperlukan pengembangan model klasifikasi prediktif yang mampu mengantisipasi status akademik mahasiswa berdasarkan data pendaftaran awal dan rekam jejak akademik mereka. Berlandaskan kebutuhan ini, kami merumuskan pertanyaan bisnis inti sebagai berikut:

1.  **Faktor-faktor apa saja yang paling signifikan memengaruhi status akademik mahasiswa (Dropout, Enrolled, Graduate) di Jaya Jaya Institute Education?**
2.  **Seberapa akurat model klasifikasi yang dikembangkan dalam memprediksi status akademik mahasiswa?**
3.  **Langkah-langkah strategis apa yang dapat diambil oleh institusi untuk mengoptimalkan tingkat retensi dan keberhasilan akademik berdasarkan wawasan dari model prediktif?**

### **Cakupan Proyek:**

Proyek ini mencakup serangkaian tahapan komprehensif, dari analisis data hingga implementasi model:

*   Analisis mendalam dan eksplorasi dataset mahasiswa.
*   Pembersihan dan persiapan data yang ketat.
*   Pembangunan, pelatihan, dan evaluasi model *machine learning* dengan optimasi *hyperparameter* menggunakan **Optuna**.
*   Visualisasi interaktif hasil analisis dan performa model menggunakan **Looker Studio**.
*   Pengembangan aplikasi prediksi *real-time* berbasis **Streamlit** untuk memfasilitasi penggunaan model oleh pihak institusi.

### **Persiapan Data**

#### **Dataset:**

Dataset yang menjadi fondasi proyek ini bersumber dari repositori GitHub, yang dapat diakses melalui tautan berikut:
**Dataset Link:** [https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

**Detail Kolom Dataset:**
-   **Marital status**: The marital status of the student.
-   **Application mode**: The method of application used by the student.
-   **Application order**: The order in which the student applied.
-   **Course**: The course taken by the student.
-   **Daytime/evening attendance**: Whether the student attends classes during the day or in the evening.
-   **Previous qualification**: The qualification obtained by the student before enrolling in higher education.
-   **Previous qualification (grade)**: Grade of previous qualification (between 0 and 200)
-   **Nacionality**: The nationality of the student.
-   **Mother's qualification**: The qualification of the student's mother.
-   **Father's qualification**: The qualification of the student's father.
-   **Mother's occupation**: The occupation of the student's mother.
-   **Father's occupation**: The occupation of the student's father.
-   **Admission grade**: Admission grade (between 0 and 200)
-   **Displaced**: Whether the student is a displaced person.
-   **Educational special needs**: Whether the student has any special educational needs.
-   **Debtor**: Whether the student is a debtor.
-   **Gender**: The gender of the student.
-   **Scholarship holder**: Whether the student is a scholarship holder.
-   **Age at enrollment**: The age of the student at the time of enrollment.
-   **International**: Whether the student is an international student.
-   **Curricular units 1st sem (credited)**: The number of curricular units credited by the student in the first semester.
-   **Curricular units 1st sem (enrolled)**: The number of curricular units enrolled by the student in the first semester.
-   **Curricular units 1st sem (evaluations)**: The number of curricular units evaluated by the student in the first semester.
-   **Curricular units 1st sem (approved)**: The number of curricular units approved by the student in the first semester.

Pastikan *environment* Anda sudah sesuai dengan `requirements.txt` sebelum melakukan *data preparation*:
```bash
pip install -r requirements.txt
```

#### **Proses Persiapan Data:**
Tahapan persiapan data dilakukan secara sistematis:
1.  **Deteksi Anomali**: Melakukan pemeriksaan awal untuk mengidentifikasi anomali atau penyimpangan dalam dataset.
2.  **Analisis Distribusi Target**: Memeriksa distribusi kolom target (`Dropout`, `Enrolled`, `Graduate`) untuk menentukan kebutuhan *data balancing* (terindikasi *imbalanced*, sehingga perlu penyeimbangan).
3.  **Encoding Kolom Target**: Mengubah kolom target menjadi representasi numerik menggunakan `LabelEncoder`.
4.  **Analisis Korelasi Fitur**: Membuat grafik *heatmap* korelasi untuk memvisualisasikan hubungan antar fitur dengan kolom target.
5.  **Pemilihan Fitur**: Memilih subset fitur yang menunjukkan korelasi signifikan dengan kolom target untuk pemodelan yang lebih efisien.
6.  **Pembagian & Skala Data**: Membagi dataset menjadi data latih (80%) dan data uji (20%), diikuti dengan proses *scaling* pada fitur untuk memastikan semua fitur berada dalam skala yang seragam.
7.  **Penyeimbangan Data Target**: Melakukan teknik *balancing* data pada kolom target yang *imbalanced* untuk mencegah bias model.

### **Modeling**

Dalam upaya membangun model prediktif yang kuat, tiga algoritma *machine learning* berbasis *ensemble* telah dilatih dan dioptimalkan menggunakan **Optuna** untuk memprediksi status akademik mahasiswa (Dropout, Enrolled, Graduate):

*   **ExtraTreesClassifier**: Metode *ensemble* yang membangun banyak *decision tree* dengan *randomness* tambahan, baik dalam pemilihan fitur maupun *splitting* data.
*   **Random Forest Classifier**: Sebuah metode *ensemble* lain yang meningkatkan akurasi prediksi dan mengurangi *overfitting* melalui agregasi beberapa *decision tree*.
*   **XGBoost (Extreme Gradient Boosting)**: Algoritma *boosting* yang sangat efisien dan performatif, dikenal karena kemampuannya dalam memberikan hasil prediksi yang akurat dengan kecepatan tinggi.

Berdasarkan evaluasi komparatif, model **XGBoostClassifier** menunjukkan performa yang paling kompetitif dan unggul di antara model-model yang dilatih ini. Model terbaik ini kemudian akan digunakan untuk membuat prediksi pada data mahasiswa di masa mendatang.

### **Evaluation**

Metrik evaluasi model yang digunakan untuk mengukur kinerja prediktif meliputi:

*   **Accuracy**: Mengukur proporsi total prediksi yang benar.
*   **Precision & Recall**: Menganalisis *trade-off* antara identifikasi positif yang akurat (Presisi) dan cakupan positif yang sebenarnya (Recall).
*   **F1-Score**: Memberikan keseimbangan antara Presisi dan Recall, sangat relevan untuk masalah klasifikasi dengan kelas yang tidak seimbang.
*   **Confusion Matrix**: Representasi visual yang detail mengenai performa model dalam mengklasifikasikan setiap kelas, membantu mengidentifikasi area misklasifikasi.

Berikut adalah ringkasan hasil evaluasi performa model yang telah dioptimalkan:

| Model                                 | Accuracy Testing | Precision (Avg) | Recall (Avg) | F1-Score (Avg) |
|---------------------------------------|------------------|-----------------|--------------|----------------|
| ExtraTrees (Optimized)                | 75.14%           | 0.73            | 0.69         | 0.69           |
| RandomForest (Optimized)              | 75.71%           | 0.71            | 0.69         | 0.70           |
| **XGBoost (Optimized)**               | **77.00%**       | **0.73**        | **0.69**     | **0.70**       |

**Detail Performa Model Terbaik (XGBoostClassifier):**

![image](https://github.com/user-attachments/assets/1172b612-65e6-4a2c-b3bf-cb0f56961682)

Model XGBoost yang telah dioptimalkan dengan *hyperparameter* terbaik (`n_estimators`: 381, `learning_rate`: 0.077, `max_depth`: 10, `min_child_weight`: 1, `subsample`: 0.664, `colsample_bytree`: 0.885) mencapai **akurasi tes sebesar 0.77 (atau 77%)**. *Confusion matrix* dan *classification report* menunjukkan performa klasifikasi status mahasiswa sebagai berikut:

*   **Graduate (Kelas 1):** Diprediksi dengan **sangat baik**. Model mengidentifikasi **409 dari 442** mahasiswa Graduate (recall **0.88**), menghasilkan F1-score tertinggi (**0.85**). Kesalahan klasifikasi minimal (**6** salah sebagai Dropout, **27** sebagai Enrolled).
*   **Dropout (Kelas 0):** Diprediksi dengan **baik**. Model mengidentifikasi **209 dari 284** mahasiswa Dropout (recall **0.70**), dengan F1-score **0.76**. Kesalahan utama adalah prediksi keliru sebagai Enrolled (**38**) dan Graduate (**37**).
*   **Enrolled (Kelas 2):** Kelas dengan prediksi **paling menantang**. Model mengidentifikasi **65 dari 159** mahasiswa Enrolled (recall **0.48**), menghasilkan F1-score **0.46**. Sering salah diklasifikasikan sebagai Graduate (**61**) dan Dropout (**33**).

Akurasi keseluruhan model adalah **0.77**. Model ini unggul dalam mengidentifikasi 'Graduate', cukup baik untuk 'Dropout', dan menunjukkan kelemahan dalam memprediksi 'Enrolled'.

---

## **Business Dashboard**

Untuk memberikan wawasan bisnis yang interaktif dan mudah diakses, sebuah *business dashboard* telah dibuat menggunakan **Looker Studio**. Dashboard ini memungkinkan pemangku kepentingan untuk menjelajahi data dan hasil analisis secara dinamis.

**Akses Dashboard Interaktif (Looker Studio):** [https://lookerstudio.google.com/reporting/8021730a-9141-49b4-8397-c12e86d1e78b](https://lookerstudio.google.com/reporting/8021730a-9141-49b4-8397-c12e86d1e78b)

## **Menjalankan Sistem Machine Learning**

Sistem *machine learning* yang telah dibangun dapat dijalankan melalui skrip Python atau diakses langsung melalui aplikasi *web* interaktif:

1.  **Menyiapkan Data**: Pastikan dataset yang diperlukan (misalnya, `data_student.csv`) tersedia di folder yang sesuai, atau siapkan data baru yang ingin Anda prediksi.
2.  **Menjalankan Model dari Skrip Python**: Untuk melakukan prediksi dengan model **XGBoost** yang telah dilatih, gunakan perintah berikut di terminal:
    ```bash
    python app.py --model xgboost_model.pkl --input data_student.csv
    ```
3.  **Akses Aplikasi Web (Streamlit)**: Untuk pengalaman *real-time* yang lebih mudah, Anda dapat mengakses aplikasi prediksi secara langsung melalui tautan berikut:
    **Aplikasi Prediksi (Streamlit):** [https://jayajaya-educational-institutions-analysis-mpfordreamer.streamlit.app/](https://jayajaya-educational-institutions-analysis-mpfordreamer.streamlit.app/)

## **Conclusion**

Proyek ini berhasil mencapai tujuan utamanya: membangun model klasifikasi yang robust untuk memprediksi status akademik mahasiswa (Dropout, Graduate, Enrolled) berdasarkan data demografi dan kinerja awal mereka.

---

**Faktor-Faktor Utama Penentu Status Akademik**
Berdasarkan analisis *feature importance* dari model **XGBoost** (lihat grafik "Feature Importance from XGBoost Model"), beberapa faktor kunci yang paling signifikan memengaruhi status akademik mahasiswa adalah:
*   `Tuition_fees_up_to_date`: Sebagai prediktor paling dominan, menyoroti pentingnya kelancaran pembayaran biaya kuliah.
*   `Curricular_units_2nd_sem_approved`: Jumlah SKS yang disetujui di semester kedua menjadi kontributor krusial kedua.
*   `Scholarship_holder`: Status sebagai penerima beasiswa juga memainkan peran penting.
*   Fitur lain seperti `Curricular_units_1st_sem_approved`, `Age_at_enrollment`, dan `Debtor` menunjukkan kontribusi yang tidak kalah penting.

---

**Model Prediktif Terbaik (XGBoostClassifier)**
Setelah optimasi *hyperparameter* yang cermat, model **XGBoostClassifier** yang terpilih menunjukkan performa yang kuat pada data uji, dengan detail sebagai berikut:
*   **Akurasi Keseluruhan**: Model mencapai akurasi sebesar **77.00%**, menjadikannya model dengan kinerja terbaik dalam proyek ini.
*   **Kinerja Berdasarkan Kelas:**
    *   **Graduate**: Diprediksi dengan **sangat baik** (Recall: **0.88**, F1-Score: **0.85**). Model berhasil mengidentifikasi **409 dari 442** mahasiswa yang benar-benar Lulus.
    *   **Dropout**: Diprediksi dengan **cukup baik** (Recall: **0.70**, F1-Score: **0.76**). Model berhasil mengidentifikasi **209 dari 284** mahasiswa yang *Dropout*.
    *   **Enrolled**: Masih menjadi kelas dengan prediksi **paling menantang** (Recall: **0.48**, F1-Score: **0.46**). Model berhasil mengidentifikasi **65 dari 159** mahasiswa yang berstatus Aktif.

Model ini terbukti sangat efektif dalam memprediksi kelulusan dan cukup andal dalam mendeteksi risiko *dropout*. Meskipun demikian, akurasi untuk memprediksi mahasiswa yang masih *Enrolled* menunjukkan ruang untuk peningkatan di masa depan.

Secara keseluruhan, proyek ini berhasil menjawab pertanyaan bisnis dan memenuhi sebagian besar tujuan yang ditetapkan. Model **XGBoostClassifier**, dengan performa prediktifnya yang kuat dan kemampuan interpretasi fitur, merupakan kandidat ideal untuk implementasi di tahap *deployment* dengan potensi peningkatan kinerja lebih lanjut.

### **Rekomendasi Aksi Strategis untuk Institusi Pendidikan**

Berdasarkan kesimpulan dan wawasan dari model prediktif, berikut adalah rekomendasi tindakan yang dapat diterapkan oleh Jaya Jaya Institute Education untuk secara signifikan menurunkan tingkat *dropout* dan meningkatkan keberhasilan akademik mahasiswa:

1.  **Dukungan Finansial Proaktif**:
    *   Implementasikan sistem deteksi dini untuk mengidentifikasi mahasiswa yang berpotensi kesulitan membayar biaya kuliah.
    *   Tawarkan opsi pembayaran yang fleksibel, program cicilan, atau bantuan keuangan darurat untuk mencegah masalah finansial menjadi penyebab *dropout*.

2.  **Perhatian Akademik di Semester Awal**:
    *   Pantau secara ketat performa akademik mahasiswa, terutama pada mata kuliah di semester pertama dan kedua.
    *   Sediakan program bimbingan belajar, sesi tutor, atau konseling akademik yang ditargetkan segera setelah terdeteksi adanya penurunan kinerja.

3.  **Perluasan dan Promosi Program Beasiswa**:
    *   Tingkatkan cakupan dan efektivitas program beasiswa, serta promosikan secara luas kepada calon mahasiswa.
    *   Beasiswa tidak hanya meringankan beban finansial tetapi juga dapat meningkatkan motivasi dan komitmen mahasiswa terhadap studi.

4.  **Manajemen Beban Studi yang Realistis**:
    *   Dorong konseling akademik yang lebih intensif untuk membantu mahasiswa merencanakan jumlah SKS yang realistis sesuai dengan kemampuan dan kesiapan mereka.
    *   Berikan dukungan ekstra bagi mahasiswa yang mengalami kesulitan dalam menyelesaikan mata kuliah yang telah diambil.

5.  **Program Mentoring dan Konseling Holistik**:
    *   Sediakan program *mentoring* dan konseling personal yang komprehensif, mempertimbangkan berbagai latar belakang mahasiswa (misalnya, usia saat pendaftaran, kebangsaan).
    *   Pendekatan ini penting untuk mengatasi tantangan adaptasi, sosial, atau psikologis yang mungkin dihadapi mahasiswa berisiko, mencegahnya berujung pada *dropout*.

---
