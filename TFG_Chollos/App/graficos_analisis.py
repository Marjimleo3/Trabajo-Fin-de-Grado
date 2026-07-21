'''
graficos_analisis.py
====================
Gráficos de análisis de precios para mostrar en el home de Streamlit.
'''

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from TFG_Chollos.utils import conseguir_ruta_general_TFG

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()
DB_ANALISIS = BASE / 'data' / 'processed' / 'analisis' / 'db_final_analisis.parquet'

MESES_ES = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

PROVINCIAS = ['Almería', 'Cádiz', 'Córdoba', 'Granada', 'Huelva', 'Jaén', 'Málaga', 'Sevilla']

BINS_TAM    = [0, 30, 50, 75, 100, 500]
LABELS_TAM  = ['<30 m2', '30-50 m2', '50-75 m2', '75-100 m2', '>100 m2']
BINS_DIST   = [0, 1, 2, 5, 10, 15]
LABELS_DIST = ['0-1 km', '1-2 km', '2-5 km', '5-10 km', '>10 km']

# =============================================================================
# FUNCIONES
# =============================================================================
@st.cache_data
def _cargar_datos() -> pd.DataFrame:
    return pd.read_parquet(DB_ANALISIS)


def _bar(x, y, title, xlabel, ylabel):
    fig = go.Figure(go.Bar(x=x, y=y, marker_color='steelblue'))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )
    return fig


def mostrar_graficos_analisis():
    st.header('Analisis de Precios')

    # Filtro de provincia en la barra lateral (vacío = todas las provincias)
    provincias_sel = st.sidebar.multiselect(
        'Filtrar por provincia:',
        options=PROVINCIAS,
    )

    df = _cargar_datos()
    if provincias_sel:
        df = df[df['provincia'].isin(provincias_sel)]

    # 1. Precio promedio por mes — línea temporal para ver estacionalidad
    df_mes = df.copy()
    df_mes['mes_num'] = df_mes['fecha_disponible'].dt.month.astype(int)
    precio_mes = df_mes.groupby('mes_num')['precio'].mean().sort_index()
    x_mes = [MESES_ES[m] for m in precio_mes.index.tolist()]
    y_mes = [round(float(v), 2) for v in precio_mes.values.tolist()]

    fig_mes = go.Figure(go.Scatter(x=x_mes, y=y_mes, mode='lines+markers', marker=dict(color='steelblue'), line=dict(color='steelblue')))
    fig_mes.update_layout(title='Precio promedio por mes', xaxis_title='Mes', yaxis_title='Precio promedio (EUR)')
    st.plotly_chart(fig_mes, width='stretch')

    col1, col2 = st.columns(2)

    # 2. Precio promedio por tamaño de habitación — agrupado en rangos de m²
    df_tam = df.copy()
    df_tam['bin'] = pd.cut(df_tam['tamaño_habitacion'].astype(float), bins=BINS_TAM, labels=False, include_lowest=True).astype('Int64')
    precio_tam = df_tam.groupby('bin')['precio'].mean().sort_index()
    x_tam = [LABELS_TAM[int(i)] for i in precio_tam.index.tolist()]
    y_tam = [round(float(v), 2) for v in precio_tam.values.tolist()]

    col1.plotly_chart(_bar(x_tam, y_tam, 'Precio promedio por tamaño de habitacion', 'Tamaño', 'Precio promedio (EUR)'), width='stretch')

    # 3. Precio promedio por distancia al centro — agrupado en rangos de km
    df_dist = df.copy()
    df_dist['bin'] = pd.cut(df_dist['distancia_centro_km'].astype(float), bins=BINS_DIST, labels=False, include_lowest=True).astype('Int64')
    precio_dist = df_dist.groupby('bin')['precio'].mean().sort_index()
    x_dist = [LABELS_DIST[int(i)] for i in precio_dist.index.tolist()]
    y_dist = [round(float(v), 2) for v in precio_dist.values.tolist()]

    col2.plotly_chart(_bar(x_dist, y_dist, 'Precio promedio por distancia al centro', 'Distancia al centro', 'Precio promedio (EUR)'), width='stretch')

    col3, col4 = st.columns(2)

    # 4. Precio promedio por tipo y periodo — barras agrupadas entre semana vs fin de semana
    df_tipo = df.copy()
    df_tipo['es_finde_int'] = df_tipo['es_finde'].astype(int)
    precio_tipo = df_tipo.groupby(['tipo', 'es_finde_int'])['precio'].mean()

    tipos = sorted(df_tipo['tipo'].unique().tolist())
    y_semana = [round(float(precio_tipo.get((t, 0), 0)), 2) for t in tipos]
    y_finde  = [round(float(precio_tipo.get((t, 1), 0)), 2) for t in tipos]

    fig4 = go.Figure([
        go.Bar(name='Entre semana', x=tipos, y=y_semana, marker_color='steelblue'),
        go.Bar(name='Fin de semana', x=tipos, y=y_finde,  marker_color='tomato'),
    ])
    fig4.update_layout(
        barmode='group',
        title='Precio promedio por tipo y periodo',
        xaxis_title='Tipo de alojamiento',
        yaxis_title='Precio promedio (EUR)',
    )
    col3.plotly_chart(fig4, width='stretch')

    # 5. Precio promedio por estrellas — relación calidad-precio por categoría hotelera
    precio_est = df.groupby('estrellas')['precio'].mean().sort_index()
    x_est = [str(int(e)) + ' estrellas' for e in precio_est.index.tolist()]
    y_est = [round(float(v), 2) for v in precio_est.values.tolist()]

    col4.plotly_chart(_bar(x_est, y_est, 'Precio promedio por estrellas', 'Estrellas', 'Precio promedio (EUR)'), width='stretch')
