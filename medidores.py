import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import re
import io
import numpy as np
from geopy.distance import geodesic
from datetime import datetime, timedelta
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache

# Configuração da página
st.set_page_config(
    page_title="Sistema de Otimização de Rotas",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurações centralizadas
CONFIG = {
    'velocidades': {'PORTO ALEGRE': 15, 'SAO LUIS': 15, 'default': 20},
    'horarios': {
        'manha_inicio': "08:30",
        'manha_fim': "12:00",
        'tarde_inicio': "14:00",
        'tarde_fim': "18:00"
    },
    'tempo_por_medidor': 15,  # minutos
    'fator_correcao': 1.3,
    'raio_filtro_km': 3,
    'aco_params': {
        'alpha': 1,
        'beta': 2,
        'evaporation': 0.5,
        'Q': 100,
        'iterations_small': 2,
        'iterations_large': 5,
        'max_ants': 20
    }
}

# CSS personalizado para melhorar a interface
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }

    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }

    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }

    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Cache otimizado para cálculo de distâncias
@lru_cache(maxsize=10000)
def calcular_distancia_cached(lat1, lon1, lat2, lon2):
    """Cache para cálculos de distância geodésica"""
    return geodesic((lat1, lon1), (lat2, lon2)).km


@st.cache_data
def load_data(file):
    """Carregamento otimizado de dados com validação"""
    try:
        with st.spinner("📊 Carregando e processando dados..."):
            df = pd.read_excel(file)
            df = df.rename(columns=lambda x: str(x).strip())

            # Validação de colunas
            required_cols = ['geometry', 'municipio']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Colunas não encontradas: {', '.join(missing_cols)}")
                return None

            # Limpeza e processamento
            df = df.dropna(subset=['geometry'])

            # Extração otimizada de coordenadas
            def extract_coords(geometry_str):
                try:
                    matches = re.findall(r"-?\d+\.\d+", str(geometry_str))
                    if len(matches) >= 2:
                        return float(matches[1]), float(matches[0])  # lat, lon
                    return None, None
                except:
                    return None, None

            coords = df['geometry'].apply(extract_coords)
            df['Latitude'] = coords.apply(lambda x: x[0])
            df['Longitude'] = coords.apply(lambda x: x[1])
            df['Município'] = df['municipio']

            # Remover coordenadas inválidas
            df = df.dropna(subset=['Latitude', 'Longitude'])

            # Validação de coordenadas
            invalid_coords = (
                    (df['Latitude'] < -90) | (df['Latitude'] > 90) |
                    (df['Longitude'] < -180) | (df['Longitude'] > 180)
            )

            if invalid_coords.any():
                st.warning(f"⚠️ {invalid_coords.sum()} coordenadas inválidas foram removidas")
                df = df[~invalid_coords]

            return df

    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
        return None


def calcular_dispersao_otimizada(df_cidade):
    """Cálculo otimizado de dispersão"""
    if len(df_cidade) == 0:
        return 0, 0

    centroide = (df_cidade['Latitude'].mean(), df_cidade['Longitude'].mean())

    # Vetorização do cálculo de distâncias
    distancias = []
    for _, row in df_cidade.iterrows():
        dist = calcular_distancia_cached(
            row['Latitude'], row['Longitude'],
            centroide[0], centroide[1]
        ) * 1000  # converter para metros
        distancias.append(dist)

    return np.mean(distancias), np.std(distancias)


def simular_rota_otimizada(df_mun, cidade, progress_callback=None):
    """Simulação de rota otimizada com feedback de progresso"""
    if len(df_mun) == 0:
        return pd.DataFrame()

    centroide = (df_mun['Latitude'].mean(), df_mun['Longitude'].mean())
    pontos_raw = df_mun[['Latitude', 'Longitude']]

    # Filtro por raio para cidades específicas
    if cidade.upper() in ['PORTO ALEGRE', 'SAO LUIS']:
        mask = pontos_raw.apply(
            lambda row: calcular_distancia_cached(
                centroide[0], centroide[1], row['Latitude'], row['Longitude']
            ) <= CONFIG['raio_filtro_km'],
            axis=1
        )
        pontos_raw = pontos_raw[mask]

    # Contagem de coordenadas
    coord_counts = pontos_raw.value_counts().to_dict()
    pontos = list(coord_counts.keys())
    n = len(pontos)

    if n == 0:
        return pd.DataFrame()

    # Parâmetros baseados na cidade
    velocidade = CONFIG['velocidades'].get(cidade.upper(), CONFIG['velocidades']['default'])
    iterations = (CONFIG['aco_params']['iterations_small']
                  if cidade.upper() in ['PORTO ALEGRE', 'SAO LUIS']
                  else CONFIG['aco_params']['iterations_large'])

    # Matriz de distâncias otimizada
    if progress_callback:
        progress_callback(0.2, "Calculando matriz de distâncias...")

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_km = calcular_distancia_cached(
                pontos[i][0], pontos[i][1],
                pontos[j][0], pontos[j][1]
            ) * CONFIG['fator_correcao']
            dist_min = (dist_km / velocidade) * 60
            dist_matrix[i][j] = dist_min
            dist_matrix[j][i] = dist_min

    # Distâncias do centroide
    centroide_dists = np.array([
        calcular_distancia_cached(centroide[0], centroide[1], pt[0], pt[1])
        * CONFIG['fator_correcao'] / velocidade * 60
        for pt in pontos
    ])

    # Algoritmo ACO otimizado
    if progress_callback:
        progress_callback(0.4, "Executando otimização ACO...")

    alpha = CONFIG['aco_params']['alpha']
    beta = CONFIG['aco_params']['beta']
    evaporation = CONFIG['aco_params']['evaporation']
    Q = CONFIG['aco_params']['Q']
    n_ants = min(CONFIG['aco_params']['max_ants'], n)

    pheromone = np.ones((n, n)) * 0.1
    best_route = None
    best_time = float('inf')

    for iteration in range(iterations):
        if progress_callback and iteration % 5 == 0:
            progress = 0.4 + (iteration / iterations) * 0.4
            progress_callback(progress, f"ACO - Iteração {iteration + 1}/{iterations}")

        iteration_best_time = float('inf')
        iteration_best_route = None

        for ant in range(n_ants):
            route = construir_rota_formiga(
                n, dist_matrix, pheromone, alpha, beta
            )

            if route and len(route) > 0:
                tempo = calcular_tempo_rota(
                    route, pontos, coord_counts, centroide_dists,
                    dist_matrix, CONFIG['tempo_por_medidor']
                )

                if tempo < iteration_best_time:
                    iteration_best_time = tempo
                    iteration_best_route = route

        if iteration_best_route and iteration_best_time < best_time:
            best_time = iteration_best_time
            best_route = iteration_best_route

        # Atualização de feromônios
        pheromone *= (1 - evaporation)
        if iteration_best_route:
            for i in range(len(iteration_best_route) - 1):
                a, b = iteration_best_route[i], iteration_best_route[i + 1]
                pheromone[a][b] += Q / iteration_best_time
                pheromone[b][a] += Q / iteration_best_time

    if not best_route:
        return pd.DataFrame()

    if progress_callback:
        progress_callback(0.9, "Gerando cronograma final...")

    # Geração do cronograma final
    rota_final = gerar_cronograma_final(
        best_route, pontos, coord_counts, centroide, velocidade
    )

    if progress_callback:
        progress_callback(1.0, "Concluído!")

    return pd.DataFrame(rota_final)


def construir_rota_formiga(n, dist_matrix, pheromone, alpha, beta):
    """Constrói rota para uma formiga individual"""
    unvisited = list(range(n))
    route = []

    if not unvisited:
        return route

    current = random.choice(unvisited)
    route.append(current)
    unvisited.remove(current)

    while unvisited:
        probs = []
        for j in unvisited:
            if dist_matrix[current][j] == 0:
                continue
            tau = pheromone[current][j] ** alpha
            eta = (1 / dist_matrix[current][j]) ** beta
            probs.append(tau * eta)

        if not probs or sum(probs) == 0:
            break

        probs = np.array(probs)
        probs = probs / probs.sum()

        try:
            next_idx = np.random.choice(len(unvisited), p=probs)
            next_city = unvisited[next_idx]
            route.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        except:
            break

    return route


def calcular_tempo_rota(route, pontos, coord_counts, centroide_dists, dist_matrix, tempo_por_medidor):
    """Calcula tempo total da rota"""
    if not route:
        return float('inf')

    tempo = centroide_dists[route[0]]  # Do centroide ao primeiro ponto

    for i in range(len(route) - 1):
        tempo += dist_matrix[route[i]][route[i + 1]]

    tempo += centroide_dists[route[-1]]  # Do último ponto ao centroide

    # Adiciona tempo de trabalho nos medidores
    tempo += sum(coord_counts[pontos[i]] for i in route) * tempo_por_medidor

    return tempo


def gerar_cronograma_final(best_route, pontos, coord_counts, centroide, velocidade):
    """Gera cronograma final com horários"""
    tempo_por_medidor = timedelta(minutes=CONFIG['tempo_por_medidor'])
    tempo_inicio_manha = datetime.strptime(CONFIG['horarios']['manha_inicio'], "%H:%M")
    tempo_fim_manha = datetime.strptime(CONFIG['horarios']['manha_fim'], "%H:%M")
    tempo_inicio_tarde = datetime.strptime(CONFIG['horarios']['tarde_inicio'], "%H:%M")
    tempo_fim_tarde = datetime.strptime(CONFIG['horarios']['tarde_fim'], "%H:%M")

    rota = []
    tempo_atual = tempo_inicio_manha
    periodo = 'manha'
    posicao_atual = centroide

    for idx in best_route:
        ponto = pontos[idx]
        dist_km = calcular_distancia_cached(
            posicao_atual[0], posicao_atual[1], ponto[0], ponto[1]
        ) * CONFIG['fator_correcao']
        tempo_desloc = timedelta(hours=dist_km / velocidade)
        qtd_medidores = coord_counts.get(tuple(ponto), 1)
        tempo_estimado = tempo_atual + tempo_desloc + (qtd_medidores * tempo_por_medidor)

        # Verificação de período
        if (periodo == 'manha' and tempo_estimado > tempo_fim_manha):
            tempo_atual = tempo_inicio_tarde
            periodo = 'tarde'
            posicao_atual = centroide
            dist_km = calcular_distancia_cached(
                posicao_atual[0], posicao_atual[1], ponto[0], ponto[1]
            ) * CONFIG['fator_correcao']
            tempo_desloc = timedelta(hours=dist_km / velocidade)
            tempo_estimado = tempo_atual + tempo_desloc + tempo_por_medidor

        if (periodo == 'tarde' and tempo_estimado > tempo_fim_tarde):
            break

        # Adiciona pontos à rota
        for i in range(qtd_medidores):
            rota.append({
                'Ordem': len(rota) + 1,
                'Latitude': ponto[0],
                'Longitude': ponto[1],
                'Horário Chegada': (tempo_atual + tempo_desloc).strftime('%H:%M'),
                'Horário Saída': (tempo_atual + tempo_desloc + tempo_por_medidor).strftime('%H:%M'),
                'Período': periodo.capitalize()
            })
            tempo_atual += tempo_por_medidor

        posicao_atual = ponto

    return rota


def processar_municipio_paralelo(args):
    """Função para processamento paralelo de municípios"""
    municipio, df_mun = args
    try:
        rota_df = simular_rota_otimizada(df_mun, municipio)
        return {
            'Município': municipio,
            'Total Medidores': len(df_mun),
            'Medidores Atendidos': len(rota_df),
            'Taxa Atendimento': len(rota_df) / len(df_mun) * 100 if len(df_mun) > 0 else 0
        }
    except Exception as e:
        return {
            'Município': municipio,
            'Total Medidores': len(df_mun),
            'Medidores Atendidos': 0,
            'Taxa Atendimento': 0,
            'Erro': str(e)
        }


# Interface principal
st.markdown('<div class="main-header"><h1>🚚 Sistema de Otimização de Rotas</h1></div>', unsafe_allow_html=True)

# Sidebar melhorada
with st.sidebar:
    st.header("📊 Navegação")
    pagina = st.radio(
        "Selecione uma opção:",
        [
            "🗺️ Mapa por Município",
            "📈 Análise de Dispersão",
            "🚚 Simulação de Rota por Município",
            "📋 Exportar Resumo da Simulação"
        ],
        index=0
    )

    st.markdown("---")
    st.header("📁 Upload de Dados")
    uploaded_file = st.file_uploader(
        "Anexe a planilha Excel",
        type="xlsx",
        help="Arquivo deve conter colunas 'municipio' e 'geometry'"
    )

    if uploaded_file:
        st.success("✅ Arquivo carregado com sucesso!")

# Processamento principal
if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        municipios = sorted(df['Município'].unique())

        # Métricas gerais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏢 Total de Municípios", len(municipios))
        with col2:
            st.metric("📍 Total de Medidores", len(df))
        with col3:
            st.metric("🎯 Média por Município", f"{len(df) / len(municipios):.0f}")
        with col4:
            maior_municipio = df['Município'].value_counts().index[0]
            st.metric("🏆 Maior Município", maior_municipio)

        st.markdown("---")

        # Páginas
        if pagina == "🚚 Simulação de Rota por Município":
            st.header("🚚 Simulação de Rota por Município")

            col1, col2 = st.columns([2, 1])
            with col1:
                selected_city = st.selectbox("Selecione um município:", municipios)
            with col2:
                st.info(f"📊 {len(df[df['Município'] == selected_city])} medidores")

            if st.button("🚀 Executar Simulação", type="primary"):
                df_mun = df[df['Município'] == selected_city]

                # Barra de progresso
                progress_bar = st.progress(0)
                status_text = st.empty()


                def update_progress(value, message):
                    progress_bar.progress(value)
                    status_text.text(message)


                start_time = time.time()
                rota_df = simular_rota_otimizada(df_mun, selected_city, update_progress)
                end_time = time.time()

                progress_bar.empty()
                status_text.empty()

                if rota_df.empty:
                    st.warning("⚠️ Nenhuma rota pôde ser gerada para os critérios definidos.")
                else:
                    st.markdown('<div class="success-message">✅ Simulação concluída com sucesso!</div>',
                                unsafe_allow_html=True)

                    # Métricas da simulação
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⏱️ Tempo de Execução", f"{end_time - start_time:.1f}s")
                    with col2:
                        st.metric("🎯 Medidores Atendidos", len(rota_df))
                    with col3:
                        taxa_atendimento = (len(rota_df) / len(df_mun)) * 100
                        st.metric("📊 Taxa de Atendimento", f"{taxa_atendimento:.1f}%")
                    with col4:
                        periodos = rota_df['Período'].value_counts()
                        st.metric("🌅/🌆 Períodos", f"{len(periodos)}")

                    # Tabela de resultados
                    st.subheader("📋 Cronograma da Rota")
                    st.dataframe(rota_df, use_container_width=True)

                    # Gráfico de distribuição por período
                    if not rota_df.empty:
                        fig = px.pie(
                            rota_df,
                            names='Período',
                            title="Distribuição de Atendimentos por Período",
                            color_discrete_map={'Manha': '#667eea', 'Tarde': '#764ba2'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Download
                    output_rota = io.BytesIO()
                    with pd.ExcelWriter(output_rota, engine='openpyxl') as writer:
                        rota_df.to_excel(writer, index=False, sheet_name='Rota')

                    st.download_button(
                        label="📤 Baixar Rota Simulada",
                        data=output_rota.getvalue(),
                        file_name=f"rota_simulada_{selected_city.replace(' ', '_').lower()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )

        elif pagina == "📋 Exportar Resumo da Simulação":
            st.header("📋 Resumo Geral da Simulação")

            if st.button("🚀 Processar Todos os Municípios", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                start_time = time.time()
                resultados = []

                # Processamento com feedback visual
                for i, municipio in enumerate(municipios):
                    progress = (i + 1) / len(municipios)
                    progress_bar.progress(progress)
                    status_text.text(f"Processando: {municipio} ({i + 1}/{len(municipios)})")

                    df_mun = df[df['Município'] == municipio]
                    try:
                        rota_df = simular_rota_otimizada(df_mun, municipio)
                        resultados.append({
                            'Município': municipio,
                            'Total Medidores': len(df_mun),
                            'Medidores Atendidos': len(rota_df),
                            'Taxa Atendimento (%)': (len(rota_df) / len(df_mun)) * 100 if len(df_mun) > 0 else 0
                        })
                    except Exception as e:
                        st.error(f"Erro em {municipio}: {str(e)}")
                        resultados.append({
                            'Município': municipio,
                            'Total Medidores': len(df_mun),
                            'Medidores Atendidos': 0,
                            'Taxa Atendimento (%)': 0
                        })

                end_time = time.time()
                progress_bar.empty()
                status_text.empty()

                df_resumo = pd.DataFrame(resultados)

                # Métricas do resumo
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⏱️ Tempo Total", f"{end_time - start_time:.1f}s")
                with col2:
                    st.metric("📊 Taxa Média", f"{df_resumo['Taxa Atendimento (%)'].mean():.1f}%")
                with col3:
                    st.metric("🎯 Total Atendidos", f"{df_resumo['Medidores Atendidos'].sum():,}")
                with col4:
                    st.metric("📍 Total Medidores", f"{df_resumo['Total Medidores'].sum():,}")

                st.subheader("📊 Resultados por Município")
                st.dataframe(df_resumo, use_container_width=True)

                # Gráfico de barras
                fig = px.bar(
                    df_resumo.head(20),
                    x='Taxa Atendimento (%)',
                    y='Município',
                    title="Top 20 Municípios - Taxa de Atendimento",
                    color='Taxa Atendimento (%)',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Download
                output_summary = io.BytesIO()
                with pd.ExcelWriter(output_summary, engine='openpyxl') as writer:
                    df_resumo.to_excel(writer, index=False, sheet_name='Resumo')

                st.download_button(
                    label="📤 Baixar Resumo Completo",
                    data=output_summary.getvalue(),
                    file_name="resumo_simulacao_rotas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

        elif pagina == "🗺️ Mapa por Município":
            st.header("🗺️ Visualizador de Coordenadas por Município")

            selected_city = st.selectbox("Selecione um município:", municipios)
            df_mun = df[df['Município'] == selected_city]

            # Métricas do município
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📍 Total de Medidores", len(df_mun))
            with col2:
                dispersao_media, _ = calcular_dispersao_otimizada(df_mun)
                st.metric("📏 Dispersão Média", f"{dispersao_media:.0f}m")
            with col3:
                densidade = len(df_mun) / (np.pi * (dispersao_media / 1000) ** 2) if dispersao_media > 0 else 0
                st.metric("🎯 Densidade", f"{densidade:.1f} med/km²")

            # Mapa individual
            st.subheader(f"📍 Mapa Detalhado - {selected_city}")

            center_lat, center_lon = df_mun['Latitude'].mean(), df_mun['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            # Adicionar marcadores
            for _, row in df_mun.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=3,
                    color='blue',
                    fill=True,
                    popup=f"{row['Município']}<br>Lat: {row['Latitude']:.4f}<br>Lon: {row['Longitude']:.4f}"
                ).add_to(m)

            # Adicionar centroide
            folium.Marker(
                location=[center_lat, center_lon],
                popup=f"Centroide - {selected_city}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)

            st.components.v1.html(m._repr_html_(), height=500)

            # Mapa geral
            st.subheader("🗺️ Visão Geral - Todos os Municípios")
            df_grouped = df.groupby('Município').agg({
                'Latitude': 'mean',
                'Longitude': 'mean',
                'geometry': 'count'
            }).rename(columns={'geometry': 'Total'})

            mapa_geral = folium.Map(
                location=[df['Latitude'].mean(), df['Longitude'].mean()],
                zoom_start=7
            )

            for municipio, row in df_grouped.iterrows():
                cor = 'red' if row['Total'] > 1000 else 'blue'
                tamanho = min(15, max(5, row['Total'] / 100))

                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=tamanho,
                    color=cor,
                    fill=True,
                    fill_color=cor,
                    fill_opacity=0.7,
                    popup=f"<b>{municipio}</b><br>{row['Total']} medidores"
                ).add_to(mapa_geral)

            st.components.v1.html(mapa_geral._repr_html_(), height=500)

        elif pagina == "📈 Análise de Dispersão":
            st.header("📈 Análise de Dispersão das Coordenadas")

            with st.spinner("Calculando dispersões..."):
                dispersoes = {}
                progress_bar = st.progress(0)

                for i, municipio in enumerate(municipios):
                    progress = (i + 1) / len(municipios)
                    progress_bar.progress(progress)

                    df_mun = df[df['Município'] == municipio]
                    media_dist, std_dist = calcular_dispersao_otimizada(df_mun)
                    dispersoes[municipio] = {
                        'Média (m)': media_dist,
                        'Desvio Padrão (m)': std_dist,
                        'Total Medidores': len(df_mun)
                    }

                progress_bar.empty()

            df_dispersao = pd.DataFrame.from_dict(dispersoes, orient='index')
            df_dispersao.index.name = 'Município'

            # Métricas gerais
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Média Geral", f"{df_dispersao['Média (m)'].mean():.0f}m")
            with col2:
                st.metric("📈 Desvio Médio", f"{df_dispersao['Desvio Padrão (m)'].mean():.0f}m")
            with col3:
                menor_dispersao = df_dispersao['Média (m)'].idxmin()
                st.metric("🎯 Menor Dispersão", menor_dispersao)
            with col4:
                maior_dispersao = df_dispersao['Média (m)'].idxmax()
                st.metric("📏 Maior Dispersão", maior_dispersao)

            # Tabela de resultados
            st.subheader("📊 Dados de Dispersão por Município")
            df_display = df_dispersao.round(0).astype({'Média (m)': int, 'Desvio Padrão (m)': int})
            st.dataframe(df_display, use_container_width=True)

            # Gráficos
            col1, col2 = st.columns(2)

            with col1:
                # Gráfico de dispersão vs total de medidores
                fig_scatter = px.scatter(
                    df_dispersao.reset_index(),
                    x='Total Medidores',
                    y='Média (m)',
                    hover_name='Município',
                    title="Dispersão vs Total de Medidores",
                    color='Desvio Padrão (m)',
                    size='Total Medidores',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col2:
                # Top 15 municípios com maior dispersão
                top_dispersao = df_dispersao.nlargest(15, 'Média (m)')
                fig_bar = px.bar(
                    top_dispersao.reset_index(),
                    x='Média (m)',
                    y='Município',
                    title="Top 15 - Maior Dispersão",
                    color='Média (m)',
                    color_continuous_scale='reds'
                )
                fig_bar.update_layout(height=500)
                st.plotly_chart(fig_bar, use_container_width=True)

            # Histograma de distribuição
            fig_hist = px.histogram(
                df_dispersao,
                x='Média (m)',
                nbins=30,
                title="Distribuição da Dispersão Média",
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Download dos dados
            output_dispersao = io.BytesIO()
            with pd.ExcelWriter(output_dispersao, engine='openpyxl') as writer:
                df_dispersao.to_excel(writer, sheet_name='Dispersão')

            st.download_button(
                label="📤 Baixar Análise de Dispersão",
                data=output_dispersao.getvalue(),
                file_name="analise_dispersao.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

else:
    st.info("👆 Por favor, faça upload de um arquivo Excel para começar a análise.")

    # Informações sobre o formato esperado
    with st.expander("📋 Formato do Arquivo"):
        st.markdown("""
        **Colunas obrigatórias:**
        - `municipio`: Nome do município
        - `geometry`: Coordenadas no formato string (ex: "POINT(-51.1234 -30.5678)")

        **Formato das coordenadas:**
        - As coordenadas devem estar no formato "POINT(longitude latitude)"
        - Valores negativos são aceitos
        - Exemplo: "POINT(-51.2177 -30.0346)"

        **Dicas:**
        - O arquivo deve estar no formato .xlsx
        - Remova linhas com coordenadas vazias
        - Certifique-se de que os nomes dos municípios estão corretos
        """)

    # Exemplo de dados
    st.subheader("📊 Exemplo de Dados")
    exemplo_df = pd.DataFrame({
        'municipio': ['Porto Alegre', 'Porto Alegre', 'Canoas', 'Canoas'],
        'geometry': [
            'POINT(-51.2177 -30.0346)',
            'POINT(-51.2158 -30.0392)',
            'POINT(-51.1843 -29.9177)',
            'POINT(-51.1821 -29.9203)'
        ]
    })
    st.dataframe(exemplo_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        🚚 Sistema de Otimização de Rotas v2.0 | 
        Desenvolvido com Streamlit e algoritmo ACO | 
        ⚡ Interface otimizada para performance
    </div>
    """,
    unsafe_allow_html=True
)
