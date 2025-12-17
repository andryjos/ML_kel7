import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Dashboard ML SD Purwakarta", layout="wide")

st.title("📊 Data Sekolah Dasar Negeri Perkecamatan di Purwakarta")

# 2. Fungsi Load Data
@st.cache_data
def load_data():
    # Membaca file dan melewati baris 'Tahun'
    df = pd.read_csv('Dataset Final Sekolah Dasar Dipurwakarta.csv', sep=';', skiprows=[1])
    df.columns = ['Kecamatan', 'Sekolah_21', 'Sekolah_22', 'Sekolah_23', 'Sekolah_24', 
                  'Siswa_21', 'Siswa_22', 'Siswa_23', 'Siswa_24', 
                  'Guru_21', 'Guru_22', 'Guru_23', 'Guru_24']
    df = df[df['Kecamatan'] != 'Jumlah'].reset_index(drop=True)
    df.index = df.index + 1 # Indeks mulai dari 1
    return df

try:
    df = load_data()

    # --- BAGIAN 1: TAMPILAN DATA ---
    st.subheader("(📄Data Sekolah Dasar Negeri)")
    st.dataframe(df, use_container_width=True)

    # --- BAGIAN 2: MODELING (SIDEBAR) ---
    st.sidebar.header("⚙️ Konfigurasi Model")
    k_value = st.sidebar.slider("Tentukan Jumlah Cluster (K)", 2, 5, 3)

    features = ['Sekolah_24', 'Siswa_24', 'Guru_24']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster'] = df['Cluster'] + 1 # Cluster ID mulai dari 1

    # --- BAGIAN 3: EVALUASI MODEL & CENTROIDS ---
    st.divider()
    st.subheader("🤖 Detail Modeling")
    tab1, tab2 = st.tabs(["Metode Elbow", "Pusat Cluster (Centroids)"])

    with tab1:
        inertia_list = []
        for k in range(1, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia_list.append(km.inertia_)
        
        fig_elbow, ax_elbow = plt.subplots(figsize=(10, 4))
        ax_elbow.plot(range(1, 11), inertia_list, marker='o', color='navy')
        ax_elbow.set_title('Elbow Method (Evaluasi Optimal K)')
        st.pyplot(fig_elbow)

    with tab2:
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        df_centroids = pd.DataFrame(centroids, columns=features)
        df_centroids.index = df_centroids.index + 1
        df_centroids.index.name = "ID Cluster"
        st.write("Rata-rata nilai fitur pada tiap pusat kelompok:")
        st.dataframe(df_centroids.style.highlight_max(axis=0))

    # --- BAGIAN 4: VISUALISASI ---
    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📍 Scatter Plot Cluster")
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='Siswa_24', y='Guru_24', hue='Cluster', 
                        palette='viridis', s=200, ax=ax_scatter, edgecolor='black')
        for i in range(len(df)):
            ax_scatter.text(df['Siswa_24'].iloc[i], df['Guru_24'].iloc[i], df['Kecamatan'].iloc[i], fontsize=8)
        st.pyplot(fig_scatter)

    with col2:
        st.subheader("🔥 Heatmap Karakteristik")
        cluster_summary = df.groupby('Cluster')[features].mean()
        fig_heat, ax_heat = plt.subplots()
        sns.heatmap(cluster_summary.T, annot=True, cmap='YlOrRd', fmt='.1f', ax=ax_heat)
        st.pyplot(fig_heat)

    # --- TOMBOL DOWNLOAD ---
    st.divider()
    csv = df.to_csv(index=True).encode('utf-8')
    st.download_button("📥 Download Hasil Analisis (CSV)", csv, "hasil_clustering.csv", "text/csv")

except Exception as e:
    st.error(f"Error: {e}")