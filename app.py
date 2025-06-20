import streamlit as st 
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
import os

# Load the model and scaler
try:
    model = joblib.load('model/best_model.joblib')
    scaler = joblib.load('model/scaler.joblib')
except Exception as e:
    st.error(f"🚨 Gagal memuat model/scaler: {str(e)}")
    st.stop()

# Load dataset for visualization
try:
    df = pd.read_csv('dataset/dashboard_full_predictions.csv')
except FileNotFoundError:
    st.error("🚨 Dataset tidak ditemukan. Pastikan file 'dashboard_full_predictions.csv' tersedia.")
    st.stop()

# Define ALL_FEATURES (sesuai dengan fitur yang digunakan saat training)
ALL_FEATURES = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification', 'Previous_qualification_grade',
    'Nacionality', 'Mothers_qualification', 'Fathers_qualification',
    'Mothers_occupation', 'Fathers_occupation', 'Admission_grade', 'Displaced',
    'Educational_special_needs', 'Debtor', 'Tuition_fees_up_to_date', 'Gender',
    'Scholarship_holder', 'Age_at_enrollment', 'International',
    'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate', 'Inflation_rate', 'GDP'
]

# Mapping untuk label
status_map = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}
gender_map = {0: 'Female', 1: 'Male'}

# Fungsi Home Page
def home_page():
    st.title("🎓 Student Academic Success Analysis")

    # Judul dan logo institusi
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #1f77b4;'>Jaya Jaya Institut Teknologi Indonesia</h3>
        <p style='color: #CCCCCC;'>Aplikasi ini membantu mendeteksi risiko mahasiswa Dropout sejak awal pendaftaran.</p>
    </div>
    """, unsafe_allow_html=True)

    # Gambar institusi (fallback jika tidak ditemukan)
    col1, col2 = st.columns([2, 3])
    with col1:
        logo_path = os.path.join("asset", "jayajaya.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=450, caption="Jaya Jaya Institut Teknologi Indonesia")
        else:
            st.markdown("""
            <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; border: 2px solid #777777; text-align: center;'>
                <h4>Logo Institusi Tidak Ditemukan</h4>
                <p>Mohon pastikan file 'asset/jayajaya.png' sudah diupload ke GitHub.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Statistik utama
        st.markdown("### 🔍 Ringkasan Data")
        st.metric("Total Mahasiswa", len(df))
        st.metric("Dropout Rate", f"{(df['Status'] == 'Dropout').mean():.1%}")
        st.metric("Graduation Rate", f"{(df['Status'] == 'Graduate').mean():.1%}")

# Load dataset for visualization
df = pd.read_csv('dataset/dashboard_full_predictions.csv')

ALL_FEATURES = [
    'Marital_status', 
    'Application_mode',
    'Application_order', 
    'Course',
    'Daytime_evening_attendance',
    'Previous_qualification',
    'Previous_qualification_grade',
    'Nacionality',
    'Mothers_qualification',
    'Fathers_qualification',
    'Mothers_occupation',
    'Fathers_occupation',
    'Admission_grade',
    'Displaced',
    'Educational_special_needs',
    'Debtor',
    'Tuition_fees_up_to_date',
    'Gender',
    'Scholarship_holder',
    'Age_at_enrollment',
    'International',
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate',
    'Inflation_rate',
    'GDP'
]

# Update definisi numerical_features dan categorical_features
numerical_features = [
    'Previous_qualification_grade',
    'Admission_grade',
    'Age_at_enrollment',
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate',
    'Inflation_rate',
    'GDP'
]

categorical_features = [
    'Application_mode',
    'Tuition_fees_up_to_date',
    'Daytime_evening_attendance',
    'Scholarship_holder',
    'Course',
    'Marital_status',
    'Previous_qualification',
    'Fathers_qualification',
    'International',
    'Displaced',
    'Application_order',
    'Gender',
    'Fathers_occupation',
    'Debtor',
    'Nacionality',
    'Educational_special_needs',
    'Mothers_qualification',
    'Mothers_occupation'
]

def home_page():
    # Title
    st.title("🎓 Student Academic Success Analysis")

    col1, col2 = st.columns([2, 2])  

    with col1:
        logo_path = os.path.join("asset", "jayajaya.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=450, caption="Jaya Jaya Institut Teknologi Indonesia")
        else:
            st.warning("⚠️ Logo institusi tidak ditemukan. Pastikan file 'asset/jayajaya.png' tersedia di direktori yang sesuai.")

    with col2:
        # Background Institution Section
        st.markdown("""
        <div style='background-color: dark; padding: 15px; border-radius: 10px;'>
            <h3>🏫 Jaya Jaya Institut Teknologi Indonesia</h3>
            <ul style='font-size: 1rem;'>
                <li>Berdiri sejak tahun 2000</li>
                <li>Mencetak lulusan berkualitas</li>
                <li>Memiliki reputasi baik dalam pendidikan tinggi</li>
                <li>Memiliki jumlah mahasiswa dropout yang meningkat</li>
                <li>Tujuan dashboard: deteksi dini, pemahaman pola, intervensi proaktif</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Key Statistics with enhanced styling
    st.markdown("### 🔑 Statistik Data")
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)  # Add spacing
    
    # Kolom statistik
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #FFFFFF88;'>
            <p style='margin: 0; font-size: 0.9rem; color: #1f77b4;'>🎓 Total Students</p>
            <h4 style='margin: 5px 0; color: #FFFFFF;'>{len(df)}</h4>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #FFFFFF88;'>
            <p style='margin: 0; font-size: 0.9rem; color: #2ca02c;'>✅ Graduation Rate</p>
            <h4 style='margin: 5px 0; color: #FFFFFF;'>{(df['Status'] == 'Graduate').mean():.1%}</h4>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #FFFFFF88;'>
            <p style='margin: 0; font-size: 0.9rem; color: #d62728;'>⚠️ Dropout Rate</p>
            <h4 style='margin: 5px 0; color: #FFFFFF;'>{(df['Status'] == 'Dropout').mean():.1%}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True) 

    # Overview Dataset Section di atas visualisasi
    st.markdown("""
    <div style='background-color: dark; padding: 20px; margin-top: 20px;'>
        <h3>📚 Overview Dataset</h3>
        <p>Dataset mencakup informasi demografi, sosial-ekonomi, latar belakang pendidikan, 
        serta kinerja akademik semester 1 & 2 dari <strong>4424 mahasiswa</strong> di berbagai jurusan.</p>
        <ul>
            <li><strong>Fitur:</strong> 37 kolom (numerik & kategorikal)</li>
            <li><strong>Target:</strong> Status → Dropout / Graduate / Enrolled</li>
            <li><strong>Tujuan:</strong> Prediksi status akhir mahasiswa sejak awal pendaftaran</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Overview visualisasi distribusi status
    fig = px.pie(df, names='Status', title='Distribusi Status Mahasiswa')
    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)


def data_insights():
    st.title("📈 Data Insights")

    # Overview Statistics with Gestalt's Similarity & Proximity
    st.header("Overview Statistics")
    col1, col2, col3 = st.columns(3)
    
    # Using consistent styling for similar metrics (Similarity principle)
    metric_style = """
    <div style="padding: 20px; border-radius: 10px; background-color: #1E1E1E; 
         text-align: center; margin: 5px; border: 1px solid rgba(255,255,255,0.1);">
        <h4 style="color: {color};">{title}</h4>
        <h2 style="color: white;">{value}</h2>
    </div>
    """
    
    with col1:
        st.markdown(
            metric_style.format(
                color="#FF6B6B",
                title="🔴 Dropout Rate",
                value=f"{(df['Status'] == 'Dropout').mean():.1%}"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            metric_style.format(
                color="#4CAF50",
                title="🟢 Graduate Rate",
                value=f"{(df['Status'] == 'Graduate').mean():.1%}"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            metric_style.format(
                color="#2196F3",
                title="🔵 Enrolled Rate",
                value=f"{(df['Status'] == 'Enrolled').mean():.1%}"
            ),
            unsafe_allow_html=True
        )
    
    # Student Demographics Analysis (Continuity principle)
    st.header("Student Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution by Status
        fig = px.box(df, x='Status', y='Age_at_enrollment', 
                    color='Status', 
                    title='Age Distribution by Status',
                    color_discrete_map={
                        'Dropout': '#FF6B6B',
                        'Graduate': '#4CAF50',
                        'Enrolled': '#2196F3'
                    })
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender Distribution
        gender_data = df.groupby(['Status', 'Gender']).size().unstack()
        fig = px.bar(gender_data, barmode='group',
                    title='Gender Distribution by Status',
                    color_discrete_sequence=['#FF69B4', '#4169E1'])
        st.plotly_chart(fig, use_container_width=True)

    # Social and Background Analysis
    st.header("Social and Background Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Marital Status Distribution
        marital_counts = df.groupby(['Marital_status', 'Status']).size().unstack()
        fig = px.bar(marital_counts, title='Marital Status Distribution by Status',
                    labels={'value': 'Count', 'Marital_status': 'Marital Status'},
                    color_discrete_sequence=['#FF6B6B', '#4CAF50', '#2196F3'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Application Mode Distribution
        app_mode_data = df.groupby('Application_mode')['Status'].value_counts().unstack()
        fig = px.bar(app_mode_data, title='Application Mode Distribution',
                    barmode='stack',
                    labels={'value': 'Count', 'Application_mode': 'Application Mode'},
                    color_discrete_sequence=['#FF6B6B', '#4CAF50', '#2196F3'])
        st.plotly_chart(fig, use_container_width=True)

    # Educational Background
    st.header("Educational Background")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Course Distribution
        course_data = df['Course'].value_counts().head(10)
        fig = px.pie(values=course_data.values, 
                    names=course_data.index,
                    title='Top 10 Courses Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Previous Qualification
        prev_qual_data = pd.crosstab(df['Previous_qualification'], df['Status'])
        fig = px.bar(prev_qual_data, title='Previous Qualification by Status',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Parents' Qualification Comparison
        parents_qual = pd.DataFrame({
            'Mother': df['Mothers_qualification'].value_counts(),
            'Father': df['Fathers_qualification'].value_counts()
        })
        fig = px.bar(parents_qual, 
                    title="Parents' Qualification Comparison",
                    barmode='group',
                    color_discrete_sequence=['#FF69B4', '#4169E1'])
        st.plotly_chart(fig, use_container_width=True)

    # Academic Progress Analysis
    st.header("Academic Progress Analysis")
    # Fix: Get the first column from the list
    col1, = st.columns(1) 
    
    with col1:
        # Units Credited vs Approved (1st Semester)
        fig = px.scatter(df, 
                        x='Curricular_units_1st_sem_credited',
                        y='Curricular_units_1st_sem_approved',
                        color='Status',
                        title='1st Semester: Credited vs Approved Units',
                        trendline="ols",
                        labels={'Curricular_units_1st_sem_credited': 'Credited Units',
                               'Curricular_units_1st_sem_approved': 'Approved Units'})
        st.plotly_chart(fig, use_container_width=True)

    # Economic Indicators
    st.header("Economic Indicators")
    col1, col2 = st.columns(2)
    
    with col1:
        # Economic Trends with error handling
        try:
            # Ensure numeric data and handle NaN values
            plot_data = df.copy()
            plot_data['GDP'] = pd.to_numeric(plot_data['GDP'], errors='coerce')
            plot_data['Unemployment_rate'] = pd.to_numeric(plot_data['Unemployment_rate'], errors='coerce')
            
            # Drop any rows with NaN values
            plot_data = plot_data.dropna(subset=['GDP', 'Unemployment_rate'])
            
            fig = px.scatter(
                plot_data, 
                x='GDP',
                y='Unemployment_rate',
                color='Status',
                title='Economic Indicators by Status',
                labels={
                    'GDP': 'GDP Growth',
                    'Unemployment_rate': 'Unemployment Rate'
                },
                color_discrete_map={
                    'Dropout': '#FF6B6B',
                    'Graduate': '#4CAF50',
                    'Enrolled': '#2196F3'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating Economic Trends visualization: {str(e)}")
            st.info("Please check the data format for GDP and Unemployment rate.")
    
    with col2:
        # Financial Status Overview with error handling
        try:
            financial_metrics = pd.DataFrame({
                'Metric': ['International', 'Displaced', 'Special Needs'],
                'Percentage': [
                    df['International'].astype(float).mean() * 100,
                    df['Displaced'].astype(float).mean() * 100,
                    df['Educational_special_needs'].astype(float).mean() * 100
                ]
            })
            
            fig = px.bar(
                financial_metrics, 
                x='Metric',
                y='Percentage',
                title='Student Demographics (%)',
                color='Percentage',
                color_continuous_scale='Viridis',
                text=financial_metrics['Percentage'].round(1).astype(str) + '%'
            )
            
            fig.update_traces(textposition='outside')
            fig.update_layout(
                yaxis_range=[0, 100],
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating Financial Status visualization: {str(e)}")
            st.info("Please check the data format for International, Displaced, and Educational special needs fields.")

    # Feature Importance (Figure-Ground principle)
    st.header("Feature Importance Analysis")
    
    # Load feature importance data
    feature_imp = pd.read_csv('dataset/feature_importance.csv')
    
    # Sort by importance
    feature_imp = feature_imp.sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(feature_imp, x='Importance', y='Feature',
                 orientation='h',
                 title='Feature Importance in Predicting Student Status',
                 color='Importance',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Analysis Conclusion
    st.header("💡 Kesimpulan Analisis")
    
    # Demographics
    with st.expander("📊 Kesimpulan Demografis"):
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px;'>
            <h4 style='color: #1f77b4;'>Analisis Demografis Mahasiswa</h4>
            <ul>
                <li>Mayoritas mahasiswa yang dropout berusia lebih tua dibanding rata-rata</li>
                <li>Terdapat perbedaan signifikan dalam distribusi gender antara mahasiswa yang lulus dan dropout</li>
                <li>Status pernikahan memiliki korelasi dengan tingkat kelulusan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Academics
    with st.expander("📚 Kesimpulan Performa Akademik"):
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px;'>
            <h4 style='color: #1f77b4;'>Analisis Performa Akademik</h4>
            <ul>
                <li>Nilai kualifikasi sebelumnya dan nilai masuk berkorelasi positif dengan kelulusan</li>
                <li>Mahasiswa yang lulus menunjukkan performa lebih baik di semester pertama</li>
                <li>Jumlah SKS yang diambil dan diluluskan memiliki pola yang berbeda antar status</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Financial
    with st.expander("💰 Kesimpulan Faktor Finansial"):
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px;'>
            <h4 style='color: #1f77b4;'>Analisis Faktor Finansial</h4>
            <ul>
                <li>Penerima beasiswa memiliki tingkat kelulusan yang lebih tinggi</li>
                <li>Status tunggakan berkorelasi dengan tingkat dropout</li>
                <li>Kondisi ekonomi (GDP dan tingkat pengangguran) mempengaruhi status akademik</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Recommendation
    with st.expander("🎯 Rekomendasi Tindakan"):
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px;'>
            <h4 style='color: #1f77b4;'>Rekomendasi untuk Institusi</h4>
            <ul>
                <li>Tingkatkan dukungan finansial untuk mahasiswa dengan risiko dropout tinggi</li>
                <li>Berikan perhatian khusus pada performa akademik semester pertama</li>
                <li>Implementasikan program mentoring untuk mahasiswa berisiko</li>
                <li>Evaluasi dan sesuaikan beban SKS berdasarkan kemampuan mahasiswa</li>
                <li>Tingkatkan akses terhadap program beasiswa</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Prediction_page
def prediction_page():
    """Halaman prediksi status akademik mahasiswa menggunakan model XGBoost"""
    
    st.title("🔮 Prediksi Status Akademik Mahasiswa")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
        <p style='color: #FFFFFF;'>Masukkan data mahasiswa untuk memprediksi kemungkinan status akademik (Dropout/Lulus/Mahasiswa Abadi)</p>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for better layout
    col1, col2 = st.columns(2)

    # Initialize input data dictionary
    input_data = {feature: 0 for feature in ALL_FEATURES}
    
    with col1:
        st.subheader("📚 Informasi Akademik")
        
        # Academic Information
        input_data['Previous_qualification_grade'] = st.number_input("Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 120.0)
        input_data['Admission_grade'] = st.number_input("Nilai Masuk", 0.0, 200.0, 120.0)
        
        # Semester 1x
        st.markdown("##### Data Semester 1")
        input_data['Curricular_units_1st_sem_approved'] = st.number_input("Jumlah SKS Lulus Semester 1", 0, 20, 6)
        input_data['Curricular_units_1st_sem_grade'] = st.number_input("Nilai Rata-rata Semester 1", 0.0, 20.0, 12.0)
        
        # Semester 2
        st.markdown("##### Data Semester 2")
        input_data['Curricular_units_2nd_sem_approved'] = st.number_input("Jumlah SKS Lulus Semester 2", 0, 20, 6)
        input_data['Curricular_units_2nd_sem_grade'] = st.number_input("Nilai Rata-rata Semester 2", 0.0, 20.0, 12.0)

    with col2:
        st.subheader("👤 Informasi Personal")
        
        # Data Demografis
        input_data['Age_at_enrollment'] = st.number_input("Usia Saat Mendaftar", 17, 70, 20)
        input_data['Gender'] = 1 if st.selectbox("Jenis Kelamin", ['Perempuan', 'Laki-laki']) == 'Laki-laki' else 0
        
        # Status Finansial
        st.markdown("##### Status Finansial")
        input_data['Tuition_fees_up_to_date'] = 1 if st.selectbox("Status Pembayaran SPP", ['Belum Lunas', 'Lunas']) == 'Lunas' else 0
        input_data['Scholarship_holder'] = 1 if st.selectbox("Penerima Beasiswa", ['Tidak', 'Ya']) == 'Ya' else 0
        input_data['Debtor'] = 1 if st.selectbox("Status Tunggakan", ['Tidak Ada', 'Ada']) == 'Ada' else 0

        # Automatically fill other required features with defaults
        for feature in ALL_FEATURES:
            if feature not in input_data:
                input_data[feature] = 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])[ALL_FEATURES]
    
    # Prediction button
    if st.button("Prediksi Status", use_container_width=True):
        try:
            probabilities = model.predict_proba(input_df)
            prediction = model.predict(input_df)[0]
            
            # Map status codes to Indonesian labels
            status_map = {0: 'Dropout', 1: 'Lulus', 2: 'Aktif'}
            predicted_status = status_map[prediction]
            
            # Get probabilities
            dropout_prob = probabilities[0][0]
            graduate_prob = probabilities[0][1]
            enrolled_prob = probabilities[0][2]
            
            # Display result
            st.markdown(f"""
            <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin-top: 10px;'>
                <h3 style='color: #1f77b4;'>Hasil Prediksi</h3>
                <p style='font-size: 1.2rem; color: white;'>Status Prediksi: <strong>{predicted_status}</strong></p>
                <div style='display: flex; justify-content: space-between;'>
                    <div style='background-color: #FF6B6B44; padding: 15px; border-radius: 5px; width: 30%;'>
                        <h4 style='color: #FF6B6B'>Dropout</h4>
                        <p style='font-size: 24px; color: white;'>{dropout_prob:.1%}</p>
                    </div>
                    <div style='background-color: #4CAF5044; padding: 15px; border-radius: 5px; width: 30%;'>
                        <h4 style='color: #4CAF50'>Lulus</h4>
                        <p style='font-size: 24px; color: white;'>{graduate_prob:.1%}</p>
                    </div>
                    <div style='background-color: #2196F344; padding: 15px; border-radius: 5px; width: 30%;'>
                        <h4 style='color: #2196F3'>Aktif</h4>
                        <p style='font-size: 24px; color: white;'>{enrolled_prob:.1%}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Recommendations in Indonesian
            if predicted_status == 'Dropout':
                st.warning("⚠️ Risiko DO Tinggi! Pertimbangkan untuk memberikan bantuan akademik dan finansial.")
            elif predicted_status == 'Lulus':
                st.success("✅ Potensi Kelulusan Tinggi! Pertahankan prestasi akademik.")
            else:
                st.info("ℹ️ Mahasiswa diprediksi tetap aktif. Pantau perkembangan secara berkala.")
                
        except Exception as e:
            st.error(f"Error Prediksi: {str(e)}")
            st.info("Pastikan semua field terisi dengan benar.")

def main():
    # Set page config
    st.set_page_config(
        page_title="Student Success Predictor",
        page_icon="🎓",
        layout="wide"
    )

    # Add title before tabs
    st.title("Jaya Jaya Institut Teknologi Indonesia Dashboard")

    # Create tabs (Continuity principle)
    tabs = st.tabs(["🏠 Home", "📊 Data Insights", "🔮 Predict"])
    
    with tabs[0]:
        home_page()
    with tabs[1]:
        data_insights()
    with tabs[2]:
        prediction_page()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <p>Developed by I Dewa Gede Mahesta Parawangsa | Dicoding ID: demahesta</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
