import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import re
import io
import numpy as np
from geopy.distance import geodesic

st.set_page_config(layout="wide")

# Menu de navegação
pagina = st.sidebar.radio("Navegar", ["Mapa por Município", "Análise de Dispersão"])

# Upload do arquivo Excel
uploaded_file = st.file_uploader("Anexe a planilha com as coordenadas (coluna 'municipio': Município, coluna 'geometry': Coordenadas)", type="xlsx")

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df = df.rename(columns=lambda x: str(x).strip())
    if 'geometry' not in df.columns or 'municipio' not in df.columns:
        st.error("❌ As colunas esperadas ('geometry' e 'municipio') não foram encontradas no arquivo.")
        st.stop()
    df = df.dropna(subset=['geometry'])
    df['Latitude'] = df['geometry'].apply(lambda x: float(re.findall(r"-\d+\.\d+", str(x))[1]))
    df['Longitude'] = df['geometry'].apply(lambda x: float(re.findall(r"-\d+\.\d+", str(x))[0]))
    df['Município'] = df['municipio']
    return df

# Cálculo de dispersão média (desvio padrão da distância ao centroide)
def calcular_dispersao(df_cidade):
    centroide = (df_cidade['Latitude'].mean(), df_cidade['Longitude'].mean())
    distancias = df_cidade.apply(lambda row: geodesic((row['Latitude'], row['Longitude']), centroide).meters, axis=1)
    return np.mean(distancias), np.std(distancias)

if uploaded_file:
    df = load_data(uploaded_file)
    municipios = sorted(df['Município'].unique())

    if pagina == "Mapa por Município":
        st.title("📍 Visualizador de Coordenadas por Município - RS")
        selected_city = st.selectbox("Selecione um município:", municipios)
        st.subheader(f"Mapa - {selected_city}")
        df_mun = df[df['Município'] == selected_city]
        m = folium.Map(location=[df_mun['Latitude'].mean(), df_mun['Longitude'].mean()], zoom_start=12)
        for _, row in df_mun.iterrows():
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"{row['Município']}"
            ).add_to(m)
        st.components.v1.html(m.get_root().render(), height=600, scrolling=True)

        # Mapa geral com pontos centrais por cidade
        st.subheader("🗺️ Mapa com ponto central por cidade")
        df_grouped = df.groupby('Município').agg({
            'Latitude': 'mean',
            'Longitude': 'mean',
            'geometry': 'count'
        }).rename(columns={'geometry': 'Total'})

        mapa_cidades = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=7)

        for idx, row in df_grouped.iterrows():
            cor = 'red' if row['Total'] > 1000 else 'blue'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=7,
                color=cor,
                fill=True,
                fill_color=cor,
                fill_opacity=0.8,
                popup=f"{idx} - {row['Total']} medidores"
            ).add_to(mapa_cidades)

        st.components.v1.html(mapa_cidades.get_root().render(), height=600, scrolling=True)

        # Botão para baixar arquivo
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        st.download_button(
            label="📥 Baixar planilha com coordenadas",
            data=output.getvalue(),
            file_name="resultado_com_coordenadas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    elif pagina == "Análise de Dispersão":
        st.title("📊 Análise de Dispersão das Coordenadas")
        dispersoes = {}
        for mun in municipios:
            df_mun = df[df['Município'] == mun]
            media_dist, std_dist = calcular_dispersao(df_mun)
            dispersoes[mun] = {'media': media_dist, 'std': std_dist}
        df_disp = pd.DataFrame.from_dict(dispersoes, orient='index')
        df_disp.index.name = 'Município'
        df_disp.columns = ['Média Distância (m)', 'Desvio Padrão (m)']

        st.dataframe(df_disp.round(2))
        st.markdown(f"**📌 Média Geral de Distância:** {df_disp['Média Distância (m)'].mean():.2f} m")
        st.markdown(f"**📌 Desvio Padrão Geral:** {df_disp['Desvio Padrão (m)'].mean():.2f} m")
