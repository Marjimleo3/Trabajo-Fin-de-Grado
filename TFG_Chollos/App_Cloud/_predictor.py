'''
_predictor.py
=============
Carga del modelo, predicción y UI de resultados. No es una página Streamlit.

Exporta:
    cargar_modelos()                             → Random Forest cacheado
    predecir_nuevos(df_features, df_info)        → (resultado, por_noche), sin categorizar
    categorizar_chollos(df, nivel_exigencia)     → df con la columna prediccion_chollo
    mostrar_resultados(df)                       → tabla + gráfico de categorías
    ETIQUETAS                                    → dict {int: str} de categorías
    NIVELES_EXIGENCIA                            → dict {nombre: umbrales} para el slider
'''

# =============================================================================
# IMPORTS
# =============================================================================
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from TFG_Chollos.Modelizacion.categorizacion import crear_etiqueta_chollo
from TFG_Chollos.utils import conseguir_ruta_general_TFG

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()

ETIQUETAS = {
    0: 'Inflado',
    1: 'Normal',
    2: 'Chollo',
    3: 'Super Chollo',
    4: 'Hiper Chollo',
}

# Umbrales de ratio (precio_real / precio_predicho) por nivel de exigencia.
# 'Predeterminado' coincide con los umbrales canónicos usados para entrenar el modelo.
NIVELES_EXIGENCIA = {
    'Predeterminado': {'umbral_hiper': 0.75, 'umbral_super': 0.85, 'umbral_chollo': 0.97, 'umbral_inflado': 1.03},
    'Exigente':       {'umbral_hiper': 0.65, 'umbral_super': 0.80, 'umbral_chollo': 0.90, 'umbral_inflado': 1.03},
    'Muy exigente':   {'umbral_hiper': 0.50, 'umbral_super': 0.70, 'umbral_chollo': 0.85, 'umbral_inflado': 1.03},
}


# =============================================================================
# CARGA DEL MODELO (una sola vez por sesión)
# =============================================================================

@st.cache_resource
def cargar_modelos():
    """
    Carga el Bosque Aleatorio podado a 20 árboles (ver Modelizacion/podar_bosque_aleatorio.py).

    El modelo completo (100 árboles) ocupa ~1,4 GB en memoria y crashea la app
    en el plan gratuito de Streamlit Community Cloud. Esta versión podada
    ocupa ~373 MB (R²=0,8634 en test, frente a 0,8676 del completo) y cabe
    directamente en GitHub (~59 MB comprimido), sin necesitar Hugging Face Hub.
    """
    return joblib.load(BASE / 'data' / 'models' / 'bosque_aleatorio_reg_ligero.pkl')


# =============================================================================
# PREDICCIÓN SOBRE DATOS NUEVOS DEL SCRAPER
# =============================================================================

# Usamos crear_etiqueta_chollo porque el scraper obtiene el precio real del calendario:
# la categoría se calcula del ratio precio_real / precio_predicho,
# garantizando coherencia con el ahorro mostrado al usuario.
def predecir_nuevos(df_features: pd.DataFrame, df_info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predice el precio justo para datos frescos del scraper (sin categorizar todavía).

    df_features y df_info traen una fila por cada noche de la estancia (mismas
    características salvo las variables temporales y el precio de esa noche). El
    modelo predice el precio justo de cada noche por separado y luego se agrupa por
    alojamiento sumando el precio real y el predicho de todas sus noches.

    La categorización de chollo se hace aparte, en categorizar_chollos(), para que
    cambiar el nivel de exigencia no requiera repetir el scraping ni la predicción.

    Parámetros
    ----------
    df_features : DataFrame codificado devuelto por codificar_nuevos() (una fila por noche)
    df_info     : DataFrame con titulo, url, precio y metadata (una fila por noche)

    Devuelve
    --------
    (resultado, por_noche)
    resultado : una fila por alojamiento; precio y precio_predicho son la suma de
                todas las noches, más ahorro. Sin prediccion_chollo todavía.
    por_noche : una fila por noche, con el precio_predicho individual de esa noche
                (para comparar precio real y predicho día a día).
    """
    if df_features.empty:
        return pd.DataFrame(), pd.DataFrame()

    bosque_reg = cargar_modelos()
    y_pred_reg = bosque_reg.predict(df_features)

    por_noche = df_info.copy()
    por_noche['precio_predicho'] = y_pred_reg.round(2)

    # Sumamos precio real y precio predicho de todas las noches por alojamiento
    resultado = por_noche.groupby('url_estancia', as_index=False).agg({
        'titulo':          'first',
        'provincia':       'first',
        'localidad':       'first',
        'tipo':            'first',
        'precio':          'sum',
        'precio_predicho': 'sum',
    })
    resultado['precio']          = resultado['precio'].round(0).astype(int)
    resultado['precio_predicho'] = resultado['precio_predicho'].round(2)
    resultado['ahorro']          = (resultado['precio_predicho'] - resultado['precio']).round(2)

    return resultado, por_noche


def categorizar_chollos(df: pd.DataFrame, nivel_exigencia: str = 'Predeterminado') -> pd.DataFrame:
    """
    Añade (o recalcula) la columna prediccion_chollo según el nivel de exigencia elegido.
    Al operar sobre precio/precio_predicho ya calculados, cambiar de nivel es instantáneo:
    no hace falta volver a scrapear ni a predecir.
    """
    if df.empty:
        return df

    df = df.copy()
    umbrales = NIVELES_EXIGENCIA[nivel_exigencia]
    categoria = crear_etiqueta_chollo(
        pd.Series(df['precio'].values),
        pd.Series(df['precio_predicho'].values),
        **umbrales,
    )
    df['prediccion_chollo'] = categoria.map(ETIQUETAS).values
    return df


# =============================================================================
# UI COMPARTIDA
# =============================================================================

def mostrar_resultados(df: pd.DataFrame):
    """
    Muestra la tabla de resultados ordenada por ahorro y un gráfico de tarta
    con la distribución de categorías. Reutilizable para BD y datos nuevos.
    """
    if df.empty:
        st.warning('No se encontraron alojamientos disponibles para las fechas seleccionadas.')
        return

    df_mostrar = df.sort_values('ahorro', ascending=False)
    st.write(f'Mostrando **{len(df_mostrar):,}** alojamientos (ordenados por mayor ahorro)')

    # Tabla interactiva con columnas configuradas
    st.dataframe(
        df_mostrar[[
            'titulo', 'localidad', 'tipo',
            'precio', 'precio_predicho', 'ahorro',
            'prediccion_chollo', 'url_estancia'
        ]],
        column_config={
            'titulo':            st.column_config.TextColumn('Alojamiento'),
            'localidad':         st.column_config.TextColumn('Localidad'),
            'tipo':              st.column_config.TextColumn('Tipo'),
            'precio':            st.column_config.NumberColumn('Precio Real (€ total)',  format='%d €'),
            'precio_predicho':   st.column_config.NumberColumn('Precio Justo (€ total)', format='%.2f €'),
            'ahorro':            st.column_config.NumberColumn('Ahorro (€ total)',        format='%.2f €'),
            'prediccion_chollo': st.column_config.TextColumn('Categoría'),
            'url_estancia':      st.column_config.LinkColumn('Ver en Booking'),
        },
        hide_index=True,
        width='stretch',
    )

    # Gráfico de tarta con la distribución de categorías, ordenado de mejor a peor
    conteo  = df_mostrar['prediccion_chollo'].value_counts()
    orden   = [ETIQUETAS[k] for k in sorted(ETIQUETAS, reverse=True) if ETIQUETAS[k] in conteo.index]
    colores = {
        'Inflado':      '#e74c3c',
        'Normal':       '#f39c12',
        'Chollo':       '#2ecc71',
        'Super Chollo': '#27ae60',
        'Hiper Chollo': '#1a7a45',
    }

    fig = go.Figure(go.Pie(
        labels=[o for o in orden],
        values=[conteo[o] for o in orden],
        marker_colors=[colores[o] for o in orden],
        hole=0.3,
        sort=False,
    ))
    fig.update_layout(title='Distribución de categorías', margin=dict(t=40, b=10))
    st.plotly_chart(fig, width='stretch')
