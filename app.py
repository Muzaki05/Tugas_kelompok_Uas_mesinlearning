import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Perceraian Jabar", layout="wide")

st.title("📊 Dashboard Analisis Perceraian Jawa Barat")
st.markdown("Aplikasi ini menganalisis penyebab perceraian di berbagai daerah di Jawa Barat.")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset_Perceraian_Jabar.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Sidebar - Navigasi
st.sidebar.header("Navigasi")
menu = st.sidebar.radio("Pilih Tampilan:", ["Ringkasan Data", "Visualisasi Distribusi", "Analisis Clustering"])

if menu == "Ringkasan Data":
    st.subheader("📋 Data Mentah")
    st.write(df)
    
    st.subheader("📈 Statistik Deskriptif")
    st.write(df.describe())

elif menu == "Visualisasi Distribusi":
    st.subheader("📈 Visualisasi Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie Chart Total Penyebab
        st.write("### Total Penyebab Perceraian")
        kategori = ['Ekonomi', 'Perselisihan', 'Meninggalkan Salah Satu', 'KDRT']
        total_counts = df[kategori].sum()
        
        fig1, ax1 = plt.subplots()
        ax1.pie(total_counts, labels=total_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        st.pyplot(fig1)

    with col2:
        # Histplot Total Kasus
        st.write("### Distribusi Total Kasus")
        df['Total_Kasus'] = df[kategori].sum(axis=1)
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Total_Kasus'], kde=True, color='orange', bins=10, ax=ax2)
        st.pyplot(fig2)

    # Bar Chart Per Kota
    st.write("### Total Kasus Berdasarkan Kota")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    df_sorted = df.sort_values('Total_Kasus', ascending=False)
    sns.barplot(data=df_sorted, x='nama daerah jawa barat', y='Total_Kasus', palette='coolwarm', ax=ax3)
    plt.xticks(rotation=90)
    st.pyplot(fig3)

elif menu == "Analisis Clustering":
    st.subheader("🤖 Clustering K-Means")
    
    # Persiapan Data Clustering
    features = ["Ekonomi", "Perselisihan", "Meninggalkan Salah Satu", "KDRT"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Slider untuk menentukan jumlah K
    k_input = st.slider("Pilih Jumlah Cluster (k):", min_value=2, max_value=6, value=4)
    
    kmeans = KMeans(n_clusters=k_input, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Visualisasi PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    colors = cm.tab10(range(k_input))
    
    for i in range(k_input):
        points = reduced_features[df['cluster'] == i]
        ax4.scatter(points[:, 0], points[:, 1], s=100, label=f'Cluster {i}', color=colors[i], edgecolors='k')
    
    # Centroids
    centroids = pca.transform(kmeans.cluster_centers_)
    ax4.scatter(centroids[:, 0], centroids[:, 1], s=250, c='black', marker='X', label='Centroids')
    
    ax4.set_title(f"Clustering dengan k={k_input}")
    ax4.legend()
    st.pyplot(fig4)
    
    st.write("### Hasil Pengelompokan Daerah:")
    st.write(df[['nama daerah jawa barat', 'cluster']].sort_values('cluster'))