import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Perceraian Jawa Barat",
    page_icon="📊",
    layout="wide"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset_Perceraian_Jabar.csv')
    df.columns = df.columns.str.strip()
    # Hitung Total Kasus secara otomatis
    df['Total_Kasus'] = df[['Ekonomi', 'Perselisihan', 'Meninggalkan Salah Satu', 'KDRT']].sum(axis=1)
    return df

# Memuat data
try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat file CSV. Pastikan file 'Dataset_Perceraian_Jabar.csv' ada di folder yang sama.")
    st.stop()

# --- 3. SIDEBAR NAVIGASI ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/b2/Logo_Jawa_Barat.png", width=100)
st.sidebar.title("Menu Utama")
menu = st.sidebar.radio(
    "Pilih Halaman:", 
    ["🏠 Beranda", "📋 Ringkasan Data", "📊 Visualisasi Distribusi", "🎯 Analisis Clustering"]
)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard ini dibuat untuk menganalisis faktor penyebab perceraian di Provinsi Jawa Barat menggunakan Machine Learning.")

# --- 4. HALAMAN: BERANDA ---
if menu == "🏠 Beranda":
    st.title("📂 Analisis Data Perceraian Provinsi Jawa Barat")
    st.image("https://images.unsplash.com/photo-1551288049-bbbda540d3b9?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80")
    
    st.markdown("""
    ### Selamat Datang di Dashboard Analisis
    Dashboard ini menyajikan data statistik mengenai angka perceraian di berbagai daerah di Jawa Barat berdasarkan empat faktor utama:
    1. **Ekonomi**
    2. **Perselisihan**
    3. **Meninggalkan Salah Satu Pasangan**
    4. **Kekerasan Dalam Rumah Tangga (KDRT)**
    
    **Tujuan Analisis:**
    * Memberikan gambaran daerah dengan tingkat perceraian tertinggi.
    * Mengelompokkan daerah (Clustering) berdasarkan pola penyebab yang serupa.
    * Memudahkan pemerintah atau instansi terkait dalam menentukan kebijakan sosial.
    """)

# --- 5. HALAMAN: RINGKASAN DATA ---
elif menu == "📋 Ringkasan Data":
    st.header("📋 Ringkasan Statistik Data")
    
    # Metrik Utama
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Kasus Jabar", f"{df['Total_Kasus'].sum():,}")
    col2.metric("Rata-rata Per Daerah", f"{int(df['Total_Kasus'].mean())}")
    col3.metric("Kasus Ekonomi Tertinggi", df.loc[df['Ekonomi'].idxmax(), 'nama daerah jawa barat'])
    col4.metric("Daerah Terdata", len(df))

    st.markdown("---")
    st.subheader("Dataframe Lengkap")
    st.dataframe(df.style.highlight_max(axis=0, color='#ffebcc'), use_container_width=True)
    
    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

# --- 6. HALAMAN: VISUALISASI DISTRIBUSI ---
elif menu == "📊 Visualisasi Distribusi":
    st.header("📊 Distribusi Penyebab Perceraian")
    
    # Pilihan daerah untuk filter
    daerah_pilihan = st.multiselect("Pilih Daerah untuk Dibandingkan:", df['nama daerah jawa barat'].unique(), default=df['nama daerah jawa barat'].unique()[:5])
    
    filtered_df = df[df['nama daerah jawa barat'].isin(daerah_pilihan)]

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Perbandingan Total Kasus")
        fig, ax = plt.subplots()
        sns.barplot(data=filtered_df, x='nama daerah jawa barat', y='Total_Kasus', palette='coolwarm', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.write("### Komposisi Faktor Penyebab")
        # Mengambil rata-rata nasional untuk pie chart
        avg_factors = df[['Ekonomi', 'Perselisihan', 'Meninggalkan Salah Satu', 'KDRT']].mean()
        fig2, ax2 = plt.subplots()
        ax2.pie(avg_factors, labels=avg_factors.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        st.pyplot(fig2)

# --- 7. HALAMAN: ANALISIS CLUSTERING ---
elif menu == "🎯 Analisis Clustering":
    st.header("🎯 Analisis Segmentasi Wilayah (K-Means)")
    st.write("Halaman ini menggunakan algoritma K-Means untuk mengelompokkan daerah dengan karakteristik penyebab perceraian yang serupa.")

    # 1. Persiapan Data
    features = ["Ekonomi", "Perselisihan", "Meninggalkan Salah Satu", "KDRT"]
    x = df[features]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(x)

    # 2. Slider Interaktif
    k_input = st.sidebar.slider("Tentukan Jumlah Kelompok (Cluster):", 2, 6, 4)
    
    # 3. Modeling
    kmeans = KMeans(n_clusters=k_input, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)

    # 4. Visualisasi PCA (Sesuai kode UAS.ipynb)
    st.subheader("Peta Sebaran Cluster (PCA Dimensi)")
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    colors = cm.get_cmap("tab10")
    
    for i in range(k_input):
        points = reduced_features[df['cluster'] == i]
        ax3.scatter(points[:, 0], points[:, 1], s=200, label=f'Cluster {i}', 
                   color=colors(i/10), edgecolors='white', alpha=0.8)

    # Label Nama Daerah
    for j, txt in enumerate(df['nama daerah jawa barat']):
        ax3.annotate(txt, (reduced_features[j, 0], reduced_features[j, 1]), 
                    fontsize=8, xytext=(5,5), textcoords='offset points')

    # Centroids
    centroids_reduced = pca.transform(kmeans.cluster_centers_)
    ax3.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], 
               s=400, c='black', marker='X', label='Pusat Cluster (Centroid)')

    ax3.set_xlabel("Komponen Utama 1 (Karakteristik Dominan)")
    ax3.set_ylabel("Komponen Utama 2 (Karakteristik Sekunder)")
    ax3.legend()
    st.pyplot(fig3)

    # 5. Analisis Hasil Cluster
    st.subheader("📋 Anggota Kelompok (Cluster)")
    for i in range(k_input):
        with st.expander(f"Lihat Daerah di Cluster {i}"):
            daerah_list = df[df['cluster'] == i]['nama daerah jawa barat'].tolist()
            st.write(", ".join(daerah_list))
            
            # Statistik rata-rata di cluster tersebut
            st.write("**Rata-rata Kasus di Cluster Ini:**")
            st.table(df[df['cluster'] == i][features].mean().to_frame().T)

st.sidebar.markdown("---")
st.sidebar.caption("© 2024 Project UAS - Analisis Data Jabar")