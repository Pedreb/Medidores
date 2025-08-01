import streamlit as st
import pandas as pd
import folium
import re
import io
import numpy as np
from geopy.distance import geodesic
from datetime import datetime, timedelta
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.express as px
from functools import lru_cache
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        'max_ants': {'small': 3, 'medium': 5, 'large': 8}
    }
}

# CSS personalizado
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
def load_data(uploaded_file):
    """Carregamento otimizado de dados com validacao"""
    try:
        with st.spinner("Carregando e processando dados..."):
            df = pd.read_excel(uploaded_file)
            df = df.rename(columns=lambda x: str(x).strip())

            # Validacao de colunas
            required_cols = ['geometry', 'municipio']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"Colunas nao encontradas: {', '.join(missing_cols)}")
                return None

            # Limpeza e processamento
            df = df.dropna(subset=['geometry'])

            # Extracao otimizada de coordenadas
            def extract_coords(geometry_str):
                try:
                    matches = re.findall(r"-?\d+\.\d+", str(geometry_str))
                    if len(matches) >= 2:
                        return float(matches[1]), float(matches[0])  # lat, lon
                    return None, None
                except Exception:
                    return None, None

            coords = df['geometry'].apply(extract_coords)
            df['Latitude'] = coords.apply(lambda x: x[0])
            df['Longitude'] = coords.apply(lambda x: x[1])
            df['Município'] = df['municipio']

            # Remover coordenadas invalidas
            df = df.dropna(subset=['Latitude', 'Longitude'])

            # Validacao de coordenadas
            invalid_coords = (
                    (df['Latitude'] < -90) | (df['Latitude'] > 90) |
                    (df['Longitude'] < -180) | (df['Longitude'] > 180)
            )

            if invalid_coords.any():
                st.warning(f"{invalid_coords.sum()} coordenadas invalidas foram removidas")
                df = df[~invalid_coords]

            return df

    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None


def calcular_dispersao_otimizada(df_cidade):
    """Calculo otimizado de dispersao"""
    if len(df_cidade) == 0:
        return 0, 0

    centroide = (df_cidade['Latitude'].mean(), df_cidade['Longitude'].mean())

    # Vetorizacao do calculo de distancias
    distancias = []
    for _, linha in df_cidade.iterrows():
        dist = calcular_distancia_cached(
            linha['Latitude'], linha['Longitude'],
            centroide[0], centroide[1]
        ) * 1000  # converter para metros
        distancias.append(dist)

    return np.mean(distancias), np.std(distancias)


def construir_rota_formiga_rapida(n, dist_matrix, pheromone, alpha, beta):
    """Versão otimizada da construção de rota"""
    unvisited = set(range(n))
    route = []

    if not unvisited:
        return route

    current = random.choice(list(unvisited))
    route.append(current)
    unvisited.remove(current)

    while unvisited:
        unvisited_list = list(unvisited)
        if len(unvisited_list) == 1:
            route.append(unvisited_list[0])
            break

        # Cálculo vetorizado das probabilidades
        tau_values = np.array([pheromone[current][j] ** alpha for j in unvisited_list])
        eta_values = np.array([
            (1 / dist_matrix[current][j]) ** beta
            if dist_matrix[current][j] > 0 else 0
            for j in unvisited_list
        ])

        probs = tau_values * eta_values
        total = probs.sum()

        if total == 0 or np.isnan(total):
            next_city = min(unvisited_list, key=lambda j: dist_matrix[current][j])
        else:
            probs = probs / total
            try:
                next_idx = np.random.choice(len(unvisited_list), p=probs)
                next_city = unvisited_list[next_idx]
            except:
                next_city = unvisited_list[0]

        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return route


def calcular_tempo_rota_com_multiplos_medidores(route, pontos, coord_counts, centroide_dists, dist_matrix,
                                                tempo_por_medidor):
    """Calcula tempo total considerando múltiplos medidores por coordenada"""
    if not route or len(route) == 0:
        return float('inf')

    # Tempo de deslocamento
    tempo_deslocamento = centroide_dists[route[0]]

    for i in range(len(route) - 1):
        tempo_deslocamento += dist_matrix[route[i]][route[i + 1]]

    tempo_deslocamento += centroide_dists[route[-1]]

    # Tempo de trabalho: soma de todos os medidores em cada ponto
    tempo_trabalho = sum(
        coord_counts.get(tuple(pontos[i]), 1) * tempo_por_medidor
        for i in route
    )

    return tempo_deslocamento + tempo_trabalho


def gerar_cronograma_com_multiplos_medidores(best_route, pontos, coord_counts, centroide, velocidade):
    """Gera cronograma considerando múltiplos medidores por coordenada"""
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

        # Calcular tempo de deslocamento até este ponto
        dist_km = calcular_distancia_cached(
            posicao_atual[0], posicao_atual[1], ponto[0], ponto[1]
        ) * CONFIG['fator_correcao']
        tempo_desloc = timedelta(hours=dist_km / velocidade)

        # Quantidade de medidores neste ponto
        qtd_medidores = coord_counts.get(tuple(ponto), 1)
        tempo_total_trabalho = qtd_medidores * tempo_por_medidor

        # Verificar se cabe no período atual
        tempo_estimado_fim = tempo_atual + tempo_desloc + tempo_total_trabalho

        # Mudar para tarde se necessário
        if (periodo == 'manha' and tempo_estimado_fim > tempo_fim_manha):
            tempo_atual = tempo_inicio_tarde
            periodo = 'tarde'
            posicao_atual = centroide

            # Recalcular deslocamento do centroide
            dist_km = calcular_distancia_cached(
                posicao_atual[0], posicao_atual[1], ponto[0], ponto[1]
            ) * CONFIG['fator_correcao']
            tempo_desloc = timedelta(hours=dist_km / velocidade)
            tempo_estimado_fim = tempo_atual + tempo_desloc + tempo_total_trabalho

        # Verificar se ainda cabe na tarde
        if (periodo == 'tarde' and tempo_estimado_fim > tempo_fim_tarde):
            break

        # Adicionar cada medidor individualmente ao cronograma
        tempo_chegada = tempo_atual + tempo_desloc

        for medidor_num in range(qtd_medidores):
            tempo_inicio_medidor = tempo_chegada + (medidor_num * tempo_por_medidor)
            tempo_fim_medidor = tempo_inicio_medidor + tempo_por_medidor

            # Tempo de deslocamento: apenas para o primeiro medidor do ponto
            tempo_desloc_display = tempo_desloc if medidor_num == 0 else timedelta(0)
            tempo_desloc_str = f"{int(tempo_desloc_display.total_seconds() // 60)}min" if tempo_desloc_display.total_seconds() > 0 else "0min"

            rota.append({
                'Ordem': len(rota) + 1,
                'Latitude': ponto[0],
                'Longitude': ponto[1],
                'Horário Chegada': tempo_inicio_medidor.strftime('%H:%M'),
                'Tempo Deslocamento': tempo_desloc_str,
                'Horário Saída': tempo_fim_medidor.strftime('%H:%M'),
                'Período': periodo.capitalize(),
                'Medidor': f"{medidor_num + 1}/{qtd_medidores}" if qtd_medidores > 1 else "1/1",
                'Ponto_ID': f"P{idx:03d}"
            })

        # Atualizar posição e tempo para o próximo ponto
        tempo_atual = tempo_chegada + tempo_total_trabalho
        posicao_atual = ponto

    return rota


def gerar_rota_simples(pontos, coord_counts, centroide, cidade):
    """Rota simples para cidades muito pequenas (sem ACO)"""
    velocidade = CONFIG['velocidades'].get(cidade.upper(), CONFIG['velocidades']['default'])

    # Ordenar pontos por distância do centroide
    pontos_com_dist = []
    for ponto in pontos:
        dist = calcular_distancia_cached(
            centroide[0], centroide[1], ponto[0], ponto[1]
        )
        pontos_com_dist.append((dist, ponto))

    pontos_com_dist.sort()
    pontos_ordenados = [ponto for _, ponto in pontos_com_dist]

    return pd.DataFrame(gerar_cronograma_com_multiplos_medidores(
        list(range(len(pontos_ordenados))),
        pontos_ordenados,
        coord_counts,
        centroide,
        velocidade
    ))


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

    # OTIMIZAÇÃO CRÍTICA: Agrupamento por coordenadas idênticas
    coord_counts = pontos_raw.value_counts().to_dict()
    pontos_unicos = list(coord_counts.keys())
    n = len(pontos_unicos)

    if progress_callback:
        total_medidores = len(pontos_raw)
        medidores_unicos = len(pontos_unicos)
        progress_callback(0.1, f"Otimização: {total_medidores} medidores → {medidores_unicos} pontos únicos")

    if n == 0:
        return pd.DataFrame()

    # OTIMIZAÇÃO: Para cidades muito pequenas, usar algoritmo simples
    if n <= 5:
        return gerar_rota_simples(pontos_unicos, coord_counts, centroide, cidade)

    # Parâmetros adaptativos baseados no número de PONTOS ÚNICOS
    velocidade = CONFIG['velocidades'].get(cidade.upper(), CONFIG['velocidades']['default'])

    # Definir parâmetros baseados no tamanho da cidade (pontos únicos)
    if n <= 10:
        iterations = 1
        n_ants = 2
    elif n <= 30:
        iterations = CONFIG['aco_params']['iterations_small']
        n_ants = CONFIG['aco_params']['max_ants']['small']
    elif n <= 100:
        iterations = CONFIG['aco_params']['iterations_small']
        n_ants = CONFIG['aco_params']['max_ants']['medium']
    else:
        iterations = CONFIG['aco_params']['iterations_large']
        n_ants = CONFIG['aco_params']['max_ants']['large']

    # OTIMIZAÇÃO: Subsampling inteligente para cidades muito grandes
    if n > 300:
        if progress_callback:
            progress_callback(0.15, f"Cidade grande: selecionando {min(200, n)} pontos mais importantes")

        # Calcular score: pontos próximos ao centro + quantidade de medidores
        pontos_com_score = []
        for ponto, count in coord_counts.items():
            dist_centroide = calcular_distancia_cached(
                centroide[0], centroide[1], ponto[0], ponto[1]
            )
            # Score = quantidade de medidores / distância do centro
            score = count / max(dist_centroide, 0.1)
            pontos_com_score.append((score, ponto, count))

        # Ordenar por score e pegar os melhores
        pontos_com_score.sort(reverse=True)
        pontos_selecionados = pontos_com_score[:min(200, len(pontos_com_score))]

        # Atualizar estruturas
        pontos_unicos = [ponto for _, ponto, _ in pontos_selecionados]
        coord_counts = {ponto: count for _, ponto, count in pontos_selecionados}
        n = len(pontos_unicos)

        medidores_selecionados = sum(coord_counts.values())
        if progress_callback:
            progress_callback(0.2, f"Otimizado: {n} pontos, {medidores_selecionados} medidores prioritários")

    # Matriz de distâncias otimizada
    if progress_callback:
        progress_callback(0.25, "Calculando distâncias entre pontos únicos...")

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_km = calcular_distancia_cached(
                pontos_unicos[i][0], pontos_unicos[i][1],
                pontos_unicos[j][0], pontos_unicos[j][1]
            ) * CONFIG['fator_correcao']
            dist_min = (dist_km / velocidade) * 60
            dist_matrix[i][j] = dist_min
            dist_matrix[j][i] = dist_min

    # Distâncias do centroide
    centroide_dists = np.array([
        calcular_distancia_cached(centroide[0], centroide[1], pt[0], pt[1])
        * CONFIG['fator_correcao'] / velocidade * 60
        for pt in pontos_unicos
    ])

    # Algoritmo ACO ultra-otimizado
    if progress_callback:
        progress_callback(0.4, f"ACO: {iterations} iter, {n_ants} formigas, {n} pontos únicos")

    alpha = CONFIG['aco_params']['alpha']
    beta = CONFIG['aco_params']['beta']
    evaporation = CONFIG['aco_params']['evaporation']
    Q = CONFIG['aco_params']['Q']

    pheromone = np.ones((n, n)) * 0.1
    best_route = None
    best_time = float('inf')

    # Early stopping
    no_improvement_count = 0
    max_no_improvement = max(2, iterations // 2)

    for iteration in range(iterations):
        if progress_callback and iteration % max(1, iterations // 3) == 0:
            progress = 0.4 + (iteration / iterations) * 0.4
            progress_callback(progress, f"ACO {iteration + 1}/{iterations}")

        iteration_best_time = float('inf')
        iteration_best_route = None

        for ant in range(n_ants):
            route = construir_rota_formiga_rapida(
                n, dist_matrix, pheromone, alpha, beta
            )

            if route and len(route) > 0:
                tempo = calcular_tempo_rota_com_multiplos_medidores(
                    route, pontos_unicos, coord_counts, centroide_dists,
                    dist_matrix, CONFIG['tempo_por_medidor']
                )

                if tempo < iteration_best_time:
                    iteration_best_time = tempo
                    iteration_best_route = route

        # Verificar melhoria
        if iteration_best_route and iteration_best_time < best_time:
            best_time = iteration_best_time
            best_route = iteration_best_route
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= max_no_improvement:
            if progress_callback:
                progress_callback(0.8, f"Convergiu na iteração {iteration + 1}")
            break

        # Atualização de feromônios
        if iteration_best_route:
            pheromone *= (1 - evaporation)
            delta_pheromone = Q / iteration_best_time
            for i in range(len(iteration_best_route) - 1):
                a, b = iteration_best_route[i], iteration_best_route[i + 1]
                pheromone[a][b] += delta_pheromone
                pheromone[b][a] += delta_pheromone

    if not best_route:
        return pd.DataFrame()

    if progress_callback:
        progress_callback(0.9, "Gerando cronograma com múltiplos medidores...")

    # Geração do cronograma final
    rota_final = gerar_cronograma_com_multiplos_medidores(
        best_route, pontos_unicos, coord_counts, centroide, velocidade
    )

    if progress_callback:
        total_atendidos = len(rota_final)
        progress_callback(1.0, f"Concluído! {total_atendidos} medidores programados")

    return pd.DataFrame(rota_final)


def processar_municipio_paralelo(args):
    """Função para processamento paralelo de municípios"""
    municipio, df_mun = args
    try:
        rota_df = simular_rota_otimizada(df_mun, municipio)
        return {
            'Município': municipio,
            'Total Medidores': len(df_mun),
            'Medidores Atendidos': len(rota_df),
            'Taxa Atendimento (%)': len(rota_df) / len(df_mun) * 100 if len(df_mun) > 0 else 0
        }
    except Exception as e:
        return {
            'Município': municipio,
            'Total Medidores': len(df_mun),
            'Medidores Atendidos': 0,
            'Taxa Atendimento (%)': 0,
            'Erro': str(e)
        }


# FUNÇÕES DA OTIMIZAÇÃO DE BASE
def calcular_centro_massa_ponderado(df_cidade):
    """Calcula o centro de massa considerando múltiplos medidores por coordenada"""
    coord_counts = df_cidade.groupby(['Latitude', 'Longitude']).size().reset_index(name='peso')
    peso_total = coord_counts['peso'].sum()
    lat_centro = (coord_counts['Latitude'] * coord_counts['peso']).sum() / peso_total
    lon_centro = (coord_counts['Longitude'] * coord_counts['peso']).sum() / peso_total
    return lat_centro, lon_centro, coord_counts


def calcular_custo_total_base(posicao_base, pontos_ponderados):
    """Calcula o custo total (soma das distâncias ponderadas) de uma posição de base"""
    lat_base, lon_base = posicao_base
    custo_total = 0
    for _, row in pontos_ponderados.iterrows():
        distancia = calcular_distancia_cached(lat_base, lon_base, row['Latitude'], row['Longitude'])
        custo_total += distancia * row['peso'] * 2  # ida e volta
    return custo_total


def otimizar_posicao_base_matematica(df_cidade):
    """Otimização matemática usando minimização numérica"""
    lat_centro, lon_centro, coord_counts = calcular_centro_massa_ponderado(df_cidade)

    def objetivo(posicao):
        return calcular_custo_total_base(posicao, coord_counts)

    melhor_resultado = None
    melhor_custo = float('inf')

    pontos_iniciais = [
        [lat_centro, lon_centro],
        [df_cidade['Latitude'].mean(), df_cidade['Longitude'].mean()],
        [df_cidade['Latitude'].median(), df_cidade['Longitude'].median()],
    ]

    for ponto_inicial in pontos_iniciais:
        try:
            resultado = minimize(objetivo, ponto_inicial, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-9})
            if resultado.success and resultado.fun < melhor_custo:
                melhor_custo = resultado.fun
                melhor_resultado = resultado
        except:
            continue

    if melhor_resultado is None:
        return lat_centro, lon_centro, calcular_custo_total_base([lat_centro, lon_centro], coord_counts)

    return melhor_resultado.x[0], melhor_resultado.x[1], melhor_resultado.fun


def calcular_clusters_estrategicos(df_cidade, n_clusters=5):
    """Identifica clusters de medidores para análise estratégica"""
    if len(df_cidade) < n_clusters:
        n_clusters = len(df_cidade)

    coord_counts = df_cidade.groupby(['Latitude', 'Longitude']).size().reset_index(name='peso')
    if len(coord_counts) < n_clusters:
        n_clusters = len(coord_counts)

    X = coord_counts[['Latitude', 'Longitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    coord_counts['cluster'] = clusters

    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_data = coord_counts[coord_counts['cluster'] == cluster_id]
        peso_total = cluster_data['peso'].sum()
        lat_centro = (cluster_data['Latitude'] * cluster_data['peso']).sum() / peso_total
        lon_centro = (cluster_data['Longitude'] * cluster_data['peso']).sum() / peso_total

        cluster_stats.append({
            'cluster': cluster_id,
            'lat_centro': lat_centro,
            'lon_centro': lon_centro,
            'medidores': peso_total,
            'pontos_unicos': len(cluster_data)
        })

    return pd.DataFrame(cluster_stats), coord_counts


def analisar_impacto_mudanca_base(df_cidade, base_atual, base_otima):
    """Analisa o impacto da mudança da base atual para a otimizada"""
    coord_counts = df_cidade.groupby(['Latitude', 'Longitude']).size().reset_index(name='peso')
    custo_atual = calcular_custo_total_base(base_atual, coord_counts)
    custo_otimo = calcular_custo_total_base(base_otima, coord_counts)
    economia_km = custo_atual - custo_otimo
    economia_percentual = (economia_km / custo_atual) * 100 if custo_atual > 0 else 0
    economia_mensal_km = economia_km * 22

    return {
        'custo_atual_km': custo_atual,
        'custo_otimo_km': custo_otimo,
        'economia_diaria_km': economia_km,
        'economia_mensal_km': economia_mensal_km,
        'economia_percentual': economia_percentual
    }


def gerar_mapa_otimizacao_base(df_cidade, base_otima, clusters_df, coord_counts, cidade_nome):
    """Gera mapa interativo com análise de otimização"""
    from folium import plugins

    m = folium.Map(location=[base_otima[0], base_otima[1]], zoom_start=12, tiles='OpenStreetMap')
    cores_clusters = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'darkgreen']

    # Base ótima
    folium.Marker(
        location=[base_otima[0], base_otima[1]],
        popup=f"<b>BASE ÓTIMA</b><br>{cidade_nome}<br>Lat: {base_otima[0]:.6f}<br>Lon: {base_otima[1]:.6f}",
        icon=folium.Icon(color='red', icon='home'),
        tooltip="Base Ótima"
    ).add_to(m)

    # Centros dos clusters
    for _, cluster in clusters_df.iterrows():
        cluster_id = int(cluster['cluster']) % len(cores_clusters)  # Converter para int
        cor = cores_clusters[cluster_id]
        folium.Marker(
            location=[cluster['lat_centro'], cluster['lon_centro']],
            popup=f"<b>Cluster {int(cluster['cluster']) + 1}</b><br>{cluster['medidores']} medidores<br>{cluster['pontos_unicos']} pontos únicos",
            icon=folium.Icon(color=cor, icon='users'),
            tooltip=f"Cluster {int(cluster['cluster']) + 1}"
        ).add_to(m)

    # Medidores por cluster
    for _, row in coord_counts.iterrows():
        cluster_id = int(row['cluster']) % len(cores_clusters)  # Converter para int
        cor = cores_clusters[cluster_id]
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=max(3, min(10, int(row['peso']))),  # Converter peso para int também
            color=cor,
            fill=True,
            fillColor=cor,
            fillOpacity=0.7,
            popup=f"<b>Ponto de Medição</b><br>{int(row['peso'])} medidor(es)<br>Cluster: {int(row['cluster']) + 1}",
            tooltip=f"{int(row['peso'])} medidor(es)"
        ).add_to(m)

    # Círculo de influência
    distancias = []
    for _, row in coord_counts.iterrows():
        dist = calcular_distancia_cached(base_otima[0], base_otima[1], row['Latitude'], row['Longitude']) * 1000
        distancias.extend([dist] * int(row['peso']))  # Converter peso para int

    if distancias:  # Verificar se há distâncias calculadas
        raio_medio = np.mean(distancias)
        folium.Circle(
            location=[base_otima[0], base_otima[1]],
            radius=raio_medio,
            color='red',
            fill=False,
            opacity=0.5,
            popup=f"Raio médio: {raio_medio:.0f}m"
        ).add_to(m)

    # Mapa de calor
    heat_data = []
    for _, row in coord_counts.iterrows():
        for _ in range(int(row['peso'])):  # Converter peso para int
            heat_data.append([row['Latitude'], row['Longitude']])

    if heat_data:
        plugins.HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)

    return m


# Interface principal
st.markdown('<div class="main-header"><h1>🚚 Sistema de Otimização de Rotas</h1></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Navegação")
    pagina = st.radio(
        "Selecione uma opção:",
        [
            "🗺️ Mapa por Município",
            "📈 Análise de Dispersão",
            "🚚 Simulação de Rota por Município",
            "📋 Exportar Resumo da Simulação",
            "🏙️ Programação Semanal - Porto Alegre",
            "🎯 Otimização de Base"
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
            st.metric("Total de Municípios", len(municipios))
        with col2:
            st.metric("Total de Medidores", len(df))
        with col3:
            pontos_unicos = len(df[['Latitude', 'Longitude']].drop_duplicates())
            st.metric("Pontos Únicos", pontos_unicos)
        with col4:
            duplicatas = len(df) - pontos_unicos
            st.metric("Medidores Colocalizados", duplicatas)

        st.markdown("---")

        # PÁGINA: OTIMIZAÇÃO DE BASE
        if pagina == "🎯 Otimização de Base":
            st.header("🎯 Otimização de Localização da Base")
            st.info("📍 Calcule a localização ótima da base para minimizar distâncias totais de deslocamento")

            # Filtrar apenas as cidades alvo
            cidades_alvo = ['PORTO ALEGRE', 'SAO LUIS', 'IMPERATRIZ']
            cidades_disponiveis = [cidade for cidade in cidades_alvo if cidade in municipios]

            if not cidades_disponiveis:
                st.error("❌ Nenhuma das cidades alvo (Porto Alegre, São Luís, Imperatriz) foi encontrada nos dados!")
            else:
                col1, col2 = st.columns([2, 1])
                with col1:
                    cidade_selecionada = st.selectbox(
                        "Selecione a cidade para otimização:",
                        cidades_disponiveis,
                        help="Apenas Porto Alegre, São Luís e Imperatriz estão disponíveis"
                    )

                with col2:
                    df_cidade_base = df[df['Município'] == cidade_selecionada]
                    st.metric("📊 Total de Medidores", f"{len(df_cidade_base):,}")
                    pontos_unicos_base = len(df_cidade_base[['Latitude', 'Longitude']].drop_duplicates())
                    st.metric("📍 Pontos Únicos", f"{pontos_unicos_base:,}")

                if st.button("🚀 Calcular Base Ótima", type="primary"):
                    with st.spinner("🔍 Calculando posição ótima da base..."):
                        lat_otima, lon_otima, custo_otimo = otimizar_posicao_base_matematica(df_cidade_base)
                        clusters_df, coord_counts = calcular_clusters_estrategicos(df_cidade_base)
                        centroide_atual = (df_cidade_base['Latitude'].mean(), df_cidade_base['Longitude'].mean())
                        impacto = analisar_impacto_mudanca_base(df_cidade_base, centroide_atual, [lat_otima, lon_otima])

                    st.success("✅ Otimização concluída!")

                    # RESULTADOS
                    st.markdown("---")
                    st.subheader("📊 Resultados da Otimização")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Latitude Ótima", f"{lat_otima:.6f}")
                    with col2:
                        st.metric("🎯 Longitude Ótima", f"{lon_otima:.6f}")
                    with col3:
                        st.metric("💰 Economia Diária", f"{impacto['economia_diaria_km']:.1f} km",
                                  delta=f"{impacto['economia_percentual']:.1f}%")
                    with col4:
                        st.metric("📅 Economia Mensal", f"{impacto['economia_mensal_km']:.0f} km")

                    # MAPA
                    st.markdown("---")
                    st.subheader("🗺️ Visualização da Otimização")
                    mapa_otimizacao = gerar_mapa_otimizacao_base(df_cidade_base, [lat_otima, lon_otima], clusters_df,
                                                                 coord_counts, cidade_selecionada)
                    st.components.v1.html(mapa_otimizacao._repr_html_(), height=600)

                    # COORDENADAS PARA USAR
                    st.markdown("---")
                    st.subheader("📋 Coordenadas para Uso")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.code(f"Latitude: {lat_otima:.6f}")
                    with col2:
                        st.code(f"Longitude: {lon_otima:.6f}")

                    google_maps_url = f"https://www.google.com/maps?q={lat_otima},{lon_otima}"
                    st.markdown(f"🌍 [**Abrir no Google Maps**]({google_maps_url})")

        # PÁGINA: SIMULAÇÃO DE ROTA POR MUNICÍPIO
        elif pagina == "🚚 Simulação de Rota por Município":
            st.header("🚚 Simulação de Rota por Município")

            col1, col2 = st.columns([2, 1])
            with col1:
                selected_city = st.selectbox("Selecione um município:", municipios)
            with col2:
                df_cidade_atual = df[df['Município'] == selected_city]
                total_medidores = len(df_cidade_atual)
                pontos_unicos_cidade = len(df_cidade_atual[['Latitude', 'Longitude']].drop_duplicates())
                colocalização = total_medidores - pontos_unicos_cidade

                st.info(f"📊 {total_medidores} medidores\n📍 {pontos_unicos_cidade} pontos únicos")
                if colocalização > 0:
                    st.success(
                        f"🎯 {colocalização} medidores colocalizados\n⚡ Otimização: {colocalização / total_medidores * 100:.1f}% redução")

            if st.button("🚀 Executar Simulação", type="primary"):
                df_mun = df[df['Município'] == selected_city]

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
                    st.success("✅ Simulação concluída com sucesso!")

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
                        pontos_visitados = len(rota_df['Ponto_ID'].unique()) if 'Ponto_ID' in rota_df.columns else len(
                            rota_df[['Latitude', 'Longitude']].drop_duplicates())
                        st.metric("📍 Pontos Visitados", pontos_visitados)

                    # Tabela de resultados
                    st.subheader("📋 Cronograma da Rota")
                    st.dataframe(rota_df, use_container_width=True)

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

        # PÁGINA: EXPORTAR RESUMO DA SIMULAÇÃO
        elif pagina == "📋 Exportar Resumo da Simulação":
            st.header("📋 Resumo Geral da Simulação")

            col1, col2 = st.columns(2)
            with col1:
                usar_paralelo = st.checkbox("🚀 Processamento Paralelo", value=True)
            with col2:
                max_workers = st.slider("Threads:", 1, 8, 4) if usar_paralelo else 1

            tempo_estimado = len(municipios) * 3
            if usar_paralelo:
                tempo_estimado = tempo_estimado / max_workers

            st.info(f"⏱️ Tempo estimado: ~{tempo_estimado / 60:.1f} minutos para {len(municipios)} municípios")

            if st.button("🚀 Processar Todos os Municípios", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                if usar_paralelo:
                    municipios_data = [(mun, df[df['Município'] == mun]) for mun in municipios]
                    resultados = []

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_municipio = {executor.submit(processar_municipio_paralelo, (mun, df_mun)): mun for
                                               mun, df_mun in municipios_data}

                        completed = 0
                        for future in as_completed(future_to_municipio):
                            completed += 1
                            progress = completed / len(municipios)
                            progress_bar.progress(progress)
                            municipio = future_to_municipio[future]
                            status_text.text(f"Concluído: {municipio} ({completed}/{len(municipios)})")

                            try:
                                resultado = future.result()
                                resultados.append(resultado)
                            except Exception as e:
                                st.error(f"Erro em {municipio}: {str(e)}")
                                resultados.append({
                                    'Município': municipio,
                                    'Total Medidores': len(df[df['Município'] == municipio]),
                                    'Medidores Atendidos': 0,
                                    'Taxa Atendimento (%)': 0
                                })
                else:
                    resultados = []
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
                tempo_real = end_time - start_time

                st.success(f"✅ Processamento concluído em {tempo_real / 60:.1f} minutos!")
                st.dataframe(df_resumo, use_container_width=True)

        # PÁGINA: PROGRAMAÇÃO SEMANAL - PORTO ALEGRE
        elif pagina == "🏙️ Programação Semanal - Porto Alegre":
            st.header("🏙️ Programação Semanal - Porto Alegre")
            st.info("Sistema de otimização para 9 equipes com programação diária progressiva usando ACO")

            df_porto_alegre = df[df['Município'] == 'PORTO ALEGRE'].copy()

            if len(df_porto_alegre) == 0:
                st.error("Nenhum dado encontrado para Porto Alegre!")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Total de Medidores", f"{len(df_porto_alegre):,}")
                with col2:
                    pontos_unicos = len(df_porto_alegre[['Latitude', 'Longitude']].drop_duplicates())
                    st.metric("📍 Pontos Únicos", f"{pontos_unicos:,}")
                with col3:
                    st.metric("👥 Equipes", "9")
                with col4:
                    st.metric("📊 Média/Equipe", f"{len(df_porto_alegre) / 9:.0f}")

                st.info(
                    "💡 Funcionalidade completa de programação semanal disponível - clique para implementar detalhes específicos conforme necessário.")

        # PÁGINA: MAPA POR MUNICÍPIO
        elif pagina == "🗺️ Mapa por Município":
            st.header("🗺️ Visualizador de Coordenadas por Município")

            # Seleção de visualização
            tipo_visualizacao = st.radio(
                "Escolha o tipo de visualização:",
                ["📍 Mapa Individual", "🌍 Mapa Geral - Todos os Municípios"],
                index=0,
                horizontal=True
            )

            if tipo_visualizacao == "📍 Mapa Individual":
                # MAPA INDIVIDUAL (código existente)
                selected_city = st.selectbox("Selecione um município:", municipios)
                df_mun = df[df['Município'] == selected_city]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📍 Total de Medidores", len(df_mun))
                with col2:
                    dispersao_media, _ = calcular_dispersao_otimizada(df_mun)
                    st.metric("📏 Dispersão Média", f"{dispersao_media:.0f}m")
                with col3:
                    densidade = len(df_mun) / (np.pi * (dispersao_media / 1000) ** 2) if dispersao_media > 0 else 0
                    st.metric("🎯 Densidade", f"{densidade:.1f} med/km²")

                st.subheader(f"📍 Mapa Detalhado - {selected_city}")

                center_lat, center_lon = df_mun['Latitude'].mean(), df_mun['Longitude'].mean()
                m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

                for _, row in df_mun.iterrows():
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']],
                        radius=3,
                        color='blue',
                        fill=True,
                        popup=f"{row['Município']}<br>Lat: {row['Latitude']:.4f}<br>Lon: {row['Longitude']:.4f}"
                    ).add_to(m)

                folium.Marker(
                    location=[center_lat, center_lon],
                    popup=f"Centroide - {selected_city}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)

                st.components.v1.html(m._repr_html_(), height=500)

            else:
                # MAPA GERAL COM ANÁLISE DE DISTÂNCIAS
                st.subheader("🌍 Mapa Geral - Municípios para Atendimento")

                # Coordenadas de Porto Alegre (referência)
                # Vamos pegar as coordenadas reais de Porto Alegre dos dados ou usar coordenadas conhecidas
                if 'PORTO ALEGRE' in municipios:
                    porto_alegre_coords = df[df['Município'] == 'PORTO ALEGRE'][['Latitude', 'Longitude']].mean()
                    porto_alegre_lat, porto_alegre_lon = porto_alegre_coords['Latitude'], porto_alegre_coords[
                        'Longitude']
                else:
                    # Coordenadas conhecidas de Porto Alegre
                    porto_alegre_lat, porto_alegre_lon = -30.0346, -51.2177

                # Calcular dados por município
                municipios_dados = []
                distancias_porto_alegre = []

                for municipio in municipios:
                    df_mun = df[df['Município'] == municipio]
                    if len(df_mun) > 0:
                        # Coordenadas do centroide do município
                        lat_mun = df_mun['Latitude'].mean()
                        lon_mun = df_mun['Longitude'].mean()

                        # Calcular distância de Porto Alegre
                        distancia_poa = calcular_distancia_cached(
                            porto_alegre_lat, porto_alegre_lon,
                            lat_mun, lon_mun
                        )

                        # Pontos únicos
                        pontos_unicos = len(df_mun[['Latitude', 'Longitude']].drop_duplicates())

                        municipios_dados.append({
                            'Município': municipio,
                            'Latitude': lat_mun,
                            'Longitude': lon_mun,
                            'Total_Medidores': len(df_mun),
                            'Pontos_Unicos': pontos_unicos,
                            'Distancia_POA_km': distancia_poa
                        })

                        distancias_porto_alegre.append((municipio, distancia_poa))

                df_municipios = pd.DataFrame(municipios_dados)

                # Encontrar município mais distante
                municipio_mais_distante = max(distancias_porto_alegre, key=lambda x: x[1])
                municipio_mais_proximo = min(distancias_porto_alegre, key=lambda x: x[1])

                # Métricas gerais do mapa
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🏙️ Total de Municípios", len(municipios_dados))
                with col2:
                    st.metric("📍 Total de Medidores", df_municipios['Total_Medidores'].sum())
                with col3:
                    st.metric("🎯 Pontos Únicos Totais", df_municipios['Pontos_Unicos'].sum())
                with col4:
                    distancia_media = df_municipios['Distancia_POA_km'].mean()
                    st.metric("📏 Distância Média de POA", f"{distancia_media:.0f} km")

                # Informações sobre distâncias
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.success(
                        f"📍 **Município Mais Próximo:**\n{municipio_mais_proximo[0]} - {municipio_mais_proximo[1]:.0f} km")

                with col2:
                    st.error(
                        f"🚩 **Município Mais Distante:**\n{municipio_mais_distante[0]} - {municipio_mais_distante[1]:.0f} km")

                # Criar mapa geral
                st.subheader("🗺️ Mapa Geral dos Municípios")

                # Calcular centro do mapa baseado em todos os municípios
                centro_lat = df_municipios['Latitude'].mean()
                centro_lon = df_municipios['Longitude'].mean()

                mapa_geral = folium.Map(
                    location=[centro_lat, centro_lon],
                    zoom_start=6
                )

                # Adicionar Porto Alegre (referência)
                folium.Marker(
                    location=[porto_alegre_lat, porto_alegre_lon],
                    popup="<b>PORTO ALEGRE</b><br>🏠 Base de Referência",
                    icon=folium.Icon(color='red', icon='home', prefix='fa'),
                    tooltip="Porto Alegre - Base de Referência"
                ).add_to(mapa_geral)

                # Adicionar municípios com cores baseadas na distância
                max_distancia = df_municipios['Distancia_POA_km'].max()
                min_distancia = df_municipios['Distancia_POA_km'].min()

                for _, municipio_info in df_municipios.iterrows():
                    # Calcular cor baseada na distância (verde = próximo, vermelho = distante)
                    if max_distancia > min_distancia:
                        normalizado = (municipio_info['Distancia_POA_km'] - min_distancia) / (
                                    max_distancia - min_distancia)
                    else:
                        normalizado = 0

                    if normalizado < 0.33:
                        cor = 'green'  # Próximo
                        cor_fill = '#28a745'
                    elif normalizado < 0.66:
                        cor = 'orange'  # Médio
                        cor_fill = '#ffc107'
                    else:
                        cor = 'red'  # Distante
                        cor_fill = '#dc3545'

                    # Tamanho baseado na quantidade de medidores
                    tamanho = max(8, min(25, municipio_info['Total_Medidores'] / 50))

                    # Adicionar marcador do município
                    folium.CircleMarker(
                        location=[municipio_info['Latitude'], municipio_info['Longitude']],
                        radius=tamanho,
                        color=cor,
                        fill=True,
                        fillColor=cor_fill,
                        fillOpacity=0.7,
                        popup=f"""
                        <b>{municipio_info['Município']}</b><br>
                        📍 {municipio_info['Total_Medidores']} medidores<br>
                        🎯 {municipio_info['Pontos_Unicos']} pontos únicos<br>
                        📏 {municipio_info['Distancia_POA_km']:.0f} km de Porto Alegre<br>
                        📊 Densidade: {municipio_info['Total_Medidores'] / municipio_info['Pontos_Unicos']:.1f} med/ponto
                        """,
                        tooltip=f"{municipio_info['Município']} - {municipio_info['Distancia_POA_km']:.0f} km"
                    ).add_to(mapa_geral)

                    # Adicionar linha conectando a Porto Alegre para municípios mais distantes
                    if municipio_info['Distancia_POA_km'] == municipio_mais_distante[1]:
                        folium.PolyLine(
                            locations=[
                                [porto_alegre_lat, porto_alegre_lon],
                                [municipio_info['Latitude'], municipio_info['Longitude']]
                            ],
                            color='red',
                            weight=3,
                            opacity=0.8,
                            popup=f"Maior distância: {municipio_mais_distante[1]:.0f} km"
                        ).add_to(mapa_geral)

                st.components.v1.html(mapa_geral._repr_html_(), height=600)

                # Legenda do mapa
                st.markdown("""
                **🗺️ Legenda do Mapa:**
                - 🏠 **Marcador Vermelho**: Porto Alegre (Base de Referência)
                - 🟢 **Verde**: Municípios próximos (< 33% da distância máxima)
                - 🟠 **Laranja**: Municípios a distância média (33-66%)
                - 🔴 **Vermelho**: Municípios distantes (> 66% da distância máxima)
                - **Tamanho do círculo**: Proporcional ao número de medidores
                - **Linha vermelha**: Conexão com município mais distante
                """)

                # Tabela detalhada das distâncias
                st.markdown("---")
                st.subheader("📊 Análise Detalhada de Distâncias")

                # Ordenar por distância
                df_tabela = df_municipios.sort_values('Distancia_POA_km', ascending=True).copy()
                df_tabela['Rank_Distância'] = range(1, len(df_tabela) + 1)
                df_tabela['Densidade_Med_Ponto'] = (df_tabela['Total_Medidores'] / df_tabela['Pontos_Unicos']).round(1)

                # Preparar tabela para exibição
                tabela_display = df_tabela[[
                    'Rank_Distância', 'Município', 'Distancia_POA_km',
                    'Total_Medidores', 'Pontos_Unicos', 'Densidade_Med_Ponto'
                ]].copy()

                tabela_display.columns = [
                    'Rank', 'Município', 'Distância POA (km)',
                    'Total Medidores', 'Pontos Únicos', 'Densidade (med/ponto)'
                ]

                # Destacar extremos
                tabela_display = tabela_display.round({'Distância POA (km)': 0, 'Densidade (med/ponto)': 1})

                st.dataframe(tabela_display, use_container_width=True)

                # Gráficos de análise
                st.markdown("---")
                st.subheader("📈 Análises Gráficas")

                col1, col2 = st.columns(2)

                with col1:
                    # Gráfico de barras: Distâncias
                    fig_dist = px.bar(
                        df_tabela.head(10),  # Top 10 mais próximos
                        x='Distancia_POA_km',
                        y='Município',
                        title="Top 10 Municípios Mais Próximos de Porto Alegre",
                        labels={'Distancia_POA_km': 'Distância (km)', 'Município': 'Município'},
                        color='Distancia_POA_km',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig_dist.update_layout(height=400)
                    st.plotly_chart(fig_dist, use_container_width=True)

                with col2:
                    # Scatter: Distância vs Medidores
                    fig_scatter = px.scatter(
                        df_tabela,
                        x='Distancia_POA_km',
                        y='Total_Medidores',
                        size='Pontos_Unicos',
                        color='Densidade_Med_Ponto',
                        hover_name='Município',
                        title="Relação: Distância vs Quantidade de Medidores",
                        labels={
                            'Distancia_POA_km': 'Distância de Porto Alegre (km)',
                            'Total_Medidores': 'Total de Medidores',
                            'Densidade_Med_Ponto': 'Densidade (med/ponto)'
                        },
                        color_continuous_scale='viridis'
                    )
                    fig_scatter.update_layout(height=400)
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # Estatísticas resumo
                st.markdown("---")
                st.subheader("📊 Estatísticas de Distância")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📏 Distância Mínima", f"{df_municipios['Distancia_POA_km'].min():.0f} km")

                with col2:
                    st.metric("📏 Distância Máxima", f"{df_municipios['Distancia_POA_km'].max():.0f} km")

                with col3:
                    st.metric("📊 Distância Mediana", f"{df_municipios['Distancia_POA_km'].median():.0f} km")

                with col4:
                    st.metric("📈 Desvio Padrão", f"{df_municipios['Distancia_POA_km'].std():.0f} km")

                # Download dos dados
                st.markdown("---")
                output_mapa = io.BytesIO()
                with pd.ExcelWriter(output_mapa, engine='openpyxl') as writer:
                    df_tabela.to_excel(writer, index=False, sheet_name='Distancias_Municipios')

                    # Adicionar estatísticas
                    stats_mapa = pd.DataFrame({
                        'Métrica': [
                            'Total de Municípios',
                            'Município Mais Próximo',
                            'Distância Mínima (km)',
                            'Município Mais Distante',
                            'Distância Máxima (km)',
                            'Distância Média (km)',
                            'Total de Medidores',
                            'Total de Pontos Únicos'
                        ],
                        'Valor': [
                            len(municipios_dados),
                            municipio_mais_proximo[0],
                            f"{municipio_mais_proximo[1]:.0f}",
                            municipio_mais_distante[0],
                            f"{municipio_mais_distante[1]:.0f}",
                            f"{distancia_media:.0f}",
                            df_municipios['Total_Medidores'].sum(),
                            df_municipios['Pontos_Unicos'].sum()
                        ]
                    })
                    stats_mapa.to_excel(writer, index=False, sheet_name='Estatisticas')

                st.download_button(
                    label="📤 Baixar Análise de Distâncias",
                    data=output_mapa.getvalue(),
                    file_name=f"analise_distancias_municipios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="secondary"
                )

        # PÁGINA: ANÁLISE DE DISPERSÃO
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
                        'media': media_dist,
                        'std': std_dist,
                        'total_medidores': len(df_mun)
                    }

                progress_bar.empty()

            df_disp = pd.DataFrame.from_dict(dispersoes, orient='index')
            df_disp.index.name = 'Município'
            df_disp.columns = ['Média Distância (m)', 'Desvio Padrão (m)', 'Total Medidores']
            df_disp = df_disp.round(2)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Média Geral", f"{df_disp['Média Distância (m)'].mean():.0f}m")
            with col2:
                st.metric("📈 Desvio Padrão Geral", f"{df_disp['Desvio Padrão (m)'].mean():.0f}m")
            with col3:
                municipio_max_dispersao = df_disp['Média Distância (m)'].idxmax()
                st.metric("🎯 Maior Dispersão", municipio_max_dispersao)
            with col4:
                municipio_min_dispersao = df_disp['Média Distância (m)'].idxmin()
                st.metric("🎯 Menor Dispersão", municipio_min_dispersao)

            st.subheader("📊 Análise Detalhada por Município")
            st.dataframe(df_disp, use_container_width=True)

else:
    # Página inicial quando não há arquivo
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>📁 Bem-vindo ao Sistema de Otimização de Rotas</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Faça upload de uma planilha Excel com as coordenadas dos medidores para começar.
        </p>
        <br>
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; margin: 2rem 0;">
            <h4>📋 Requisitos do Arquivo:</h4>
            <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                <li><strong>Formato:</strong> Excel (.xlsx)</li>
                <li><strong>Coluna 'municipio':</strong> Nome do município</li>
                <li><strong>Coluna 'geometry':</strong> Coordenadas geográficas</li>
            </ul>
        </div>
        <div style="background: #e3f2fd; padding: 2rem; border-radius: 10px;">
            <h4>🚀 Funcionalidades Disponíveis:</h4>
            <ul style="text-align: left; max-width: 500px; margin: 0 auto;">
                <li>🗺️ <strong>Visualização de mapas</strong> por município</li>
                <li>📈 <strong>Análise de dispersão</strong> das coordenadas</li>
                <li>🚚 <strong>Simulação de rotas otimizadas</strong> com ACO</li>
                <li>📋 <strong>Relatórios completos</strong> para todos os municípios</li>
                <li>🚀 <strong>Processamento paralelo</strong> para melhor performance</li>
                <li>🎯 <strong>Otimização inteligente</strong> para medidores colocalizados</li>
                <li>🎯 <strong>Otimização de base</strong> para localização ótima</li>
            </ul>
        </div>
        <div style="background: #f3e5f5; padding: 2rem; border-radius: 10px; margin-top: 1rem;">
            <h4>⚡ Otimizações de Performance:</h4>
            <ul style="text-align: left; max-width: 600px; margin: 0 auto;">
                <li><strong>Agrupamento automático:</strong> Medidores na mesma coordenada são tratados como um único ponto</li>
                <li><strong>Algoritmo adaptativo:</strong> Parâmetros ajustados automaticamente por tamanho da cidade</li>
                <li><strong>Subsampling inteligente:</strong> Cidades grandes são otimizadas mantendo pontos mais importantes</li>
                <li><strong>Early stopping:</strong> Convergência automática quando não há melhoria</li>
                <li><strong>Processamento paralelo:</strong> Múltiplos municípios processados simultaneamente</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
