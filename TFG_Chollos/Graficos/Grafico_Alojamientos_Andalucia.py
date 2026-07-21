"""
Genera el mapa coroplético de alojamientos de Andalucía por provincias,
con todos los puntos únicos de alojamiento de db_final_analisis.parquet.
El HTML resultante se usa como mapa predeterminado en el Streamlit.

Uso:
    python Graficos/Grafico_Alojamientos_Andalucia.py
"""

# =============================================================================
# IMPORTS
# =============================================================================
import re

import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from TFG_Chollos.utils import conseguir_ruta_general_TFG

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()

NOMBRES_PROVINCIAS = ['Sevilla', 'Cádiz', 'Huelva', 'Jaén', 'Granada', 'Almería', 'Córdoba', 'Málaga']

CENTROIDES_PROVINCIAS = {
    'Almería':  (37.15, -2.36),
    'Cádiz':    (36.60, -5.80),
    'Córdoba':  (37.90, -4.77),
    'Granada':  (37.20, -3.40),
    'Huelva':   (37.60, -6.94),
    'Jaén':     (37.90, -3.50),
    'Málaga':   (36.80, -4.70),
    'Sevilla':  (37.50, -5.80),
}


# =============================================================================
# FUNCIONES
# =============================================================================
def scrape_n_alojamientos(urls_provincias: dict) -> dict:
    resultado = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            java_script_enabled=True,
            locale="es-ES",
            timezone_id="Europe/Madrid",
        )
        page = context.new_page()
        page.route("**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf}", lambda route: route.abort())
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        for provincia, url in urls_provincias.items():
            page.goto(url, wait_until="networkidle")
            soup = BeautifulSoup(page.content(), 'html.parser')
            titulo = soup.find('h1')
            if titulo is None:
                raise ValueError(f"No se encontró h1 para {provincia}")
            span  = titulo.find('span')
            texto = (span if span else titulo).get_text(strip=True)
            match = re.search(r"[\d.,]+", texto)
            if not match:
                raise ValueError(f"No se encontró número en: {texto!r}")
            numero = int(match.group().replace(".", "").replace(",", ""))
            resultado[provincia] = numero
            print(f"  {provincia}: {numero} alojamientos")

        browser.close()
    return resultado


def cargar_puntos_unicos() -> pd.DataFrame:
    df = pd.read_parquet(
        BASE / 'data' / 'processed' / 'analisis' / 'db_final_analisis.parquet',
        columns=['titulo', 'latitud', 'longitud', 'url_estancia']
    )
    df = df.drop_duplicates(subset='url_estancia')[['titulo', 'latitud', 'longitud']]
    # Filtrar coordenadas fuera de Andalucía (evita que el mapa se aleje)
    df = df[(df['latitud'].between(35.8, 38.7)) & (df['longitud'].between(-7.6, -1.6))]
    # Reducir densidad deduplicando por celda de ~1km (2 decimales ≈ 1.1km)
    df['_lat_r'] = df['latitud'].round(2)
    df['_lon_r'] = df['longitud'].round(2)
    df = df.drop_duplicates(subset=['_lat_r', '_lon_r'])
    return df[['titulo', 'latitud', 'longitud']]


def generar_mapa(alojamientos_por_provincia: dict, geojson: dict, df_puntos: pd.DataFrame,
                 fecha_entrada: str, fecha_salida: str):
    # Reordenamos por nombre de provincia (no por posición) para que cada valor se
    # empareje con la provincia correcta, independientemente del orden de scraping.
    valores = [alojamientos_por_provincia.get(p, 0) for p in NOMBRES_PROVINCIAS]
    maximo = max(valores)
    techo = (maximo // 1000 + 1) * 1000

    fig = px.choropleth(
        locations=NOMBRES_PROVINCIAS,
        geojson=geojson,
        featureidkey="properties.name",
        color=valores,
        range_color=[0, techo],
        color_continuous_scale="Reds",
        title=f"Alojamientos en Andalucía ({fecha_entrada} → {fecha_salida})",
        hover_name=NOMBRES_PROVINCIAS,
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Alojamientos: %{z:,.0f}<extra></extra>")
    fig.update_geos(
        visible=False,
        lataxis_range=[35.5, 39.0],
        lonaxis_range=[  -8.0, -1.0],
    )
    fig.update_layout(
        height=600,
        paper_bgcolor="white",
        margin={"r": 20, "t": 40, "l": 20, "b": 10},
        coloraxis_colorbar=dict(title="Nº Aloj. disp.", tickformat=",.0f"),
        legend=dict(
            x=0.01,
            y=0.01,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="gray",
            borderwidth=1),
    )
    fig.add_scattergeo(
        lat=[v[0] for v in CENTROIDES_PROVINCIAS.values()],
        lon=[v[1] for v in CENTROIDES_PROVINCIAS.values()],
        mode="text",
        text=list(CENTROIDES_PROVINCIAS.keys()),
        textfont=dict(size=11, color="black"),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_scattergeo(
        lat=df_puntos['latitud'],
        lon=df_puntos['longitud'],
        mode="markers",
        marker=dict(size=5, color="green", line=dict(width=1, color="white")),
        text=df_puntos['titulo'],
        hovertemplate="<b>%{text}</b><extra></extra>",
        name='Ocultar alojamientos',
    )
    return fig


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    df_urls = pd.read_csv(BASE / "data" / "raw" / "inputs" / "urls_busqueda_booking_provincias.csv", sep="|")
    urls_provincias = df_urls.set_index("localizacion")["url"].to_dict()

    with open(BASE / "data" / "raw" / "inputs" / "fecha_entrada_busqueda_booking.txt") as f:
        fecha_entrada = f.read().strip()
    with open(BASE / "data" / "raw" / "inputs" / "fecha_salida_busqueda_booking.txt") as f:
        fecha_salida = f.read().strip()

    print(f"Scrapeando conteos ({fecha_entrada} → {fecha_salida})...")
    alojamientos = scrape_n_alojamientos(urls_provincias)

    print("Cargando GeoJSON...")
    url_geojson = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/spain-provinces.geojson"
    geojson_raw = requests.get(url_geojson).json()
    geojson = {
        "type": "FeatureCollection",
        "features": [f for f in geojson_raw["features"] if f["properties"]["name"] in NOMBRES_PROVINCIAS]
    }

    print("Cargando puntos únicos de alojamientos...")
    df_puntos = cargar_puntos_unicos()
    print(f"  {len(df_puntos)} alojamientos únicos")

    fig = generar_mapa(alojamientos, geojson, df_puntos, fecha_entrada, fecha_salida)

    salida = BASE / "Graficos" / "mapa_predeterminado.html"
    fig.write_html(salida)
    print(f"[OK] Mapa guardado en: {salida}")


if __name__ == "__main__":
    main()
