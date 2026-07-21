'''
Home - Mapa de Alojamientos de Andalucía
=========================================
Para ejecutar:
    python App/run.py
    python TFG_Chollos/App/run.py
'''

# =============================================================================
# IMPORTS
# =============================================================================
import re
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from TFG_Chollos.utils import conseguir_ruta_general_TFG
from graficos_analisis import mostrar_graficos_analisis
from TFG_Chollos.Scraping.Generador_urls_generales import (
    generador_urls, PROVINCIAS, N_ADULTOS, N_HABITACIONES, N_MENORES
)
from TFG_Chollos.Graficos.Grafico_Alojamientos_Andalucia import generar_mapa, NOMBRES_PROVINCIAS


# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()
MAPA_PREDETERMINADO = BASE / 'Graficos' / 'mapa_predeterminado.html'
MAPA_ACTUALIZADO    = BASE / 'Graficos' / 'mapa_actualizado.html'


# =============================================================================
# FUNCIONES
# =============================================================================
@st.cache_data
def cargar_geojson():
    url = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/spain-provinces.geojson"
    geojson = requests.get(url).json()
    return {
        "type": "FeatureCollection",
        "features": [f for f in geojson["features"] if f["properties"]["name"] in NOMBRES_PROVINCIAS]
    }


@st.cache_data
def cargar_puntos_unicos() -> pd.DataFrame:
    """
    Carga las coordenadas únicas de cada alojamiento para pintarlos en el mapa.
    Deduplica por URL y luego por celda de ~1 km (2 decimales) para reducir densidad visual.
    Filtra coordenadas fuera de Andalucía para que el mapa no se aleje.
    """
    df = pd.read_parquet(
        BASE / 'data' / 'processed' / 'analisis' / 'db_final_analisis.parquet',
        columns=['titulo', 'latitud', 'longitud', 'url_estancia']
    )
    # Eliminamos duplicados por URL y filtramos coordenadas fuera de Andalucía
    df = df.drop_duplicates(subset='url_estancia')[['titulo', 'latitud', 'longitud']]
    df = df[(df['latitud'].between(35.8, 38.7)) & (df['longitud'].between(-7.6, -1.6))]
    # Reducimos densidad deduplicando por celda de ~1 km (2 decimales ≈ 1.1 km)
    df['_lat_r'] = df['latitud'].round(2)
    df['_lon_r'] = df['longitud'].round(2)
    df = df.drop_duplicates(subset=['_lat_r', '_lon_r'])
    return df[['titulo', 'latitud', 'longitud']]


async def _async_scrape(urls_provincias: dict, resultado: dict, progreso: list):
    """
    Navega con Playwright a la página de resultados de cada provincia y extrae
    el número total de alojamientos disponibles del h1. Bloquea recursos estáticos
    para acelerar la carga y oculta la huella de automatización.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--disable-blink-features=AutomationControlled"])
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            java_script_enabled=True,
            locale="es-ES",
            timezone_id="Europe/Madrid",
        )
        page = await context.new_page()

        # Bloqueamos imágenes y fuentes para acelerar la carga de cada página
        async def _abort(route):
            await route.abort()
        await page.route("**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf}", _abort)
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        provincias_lista = list(urls_provincias.items())
        for i, (provincia, url) in enumerate(provincias_lista):
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_selector('h1', state='visible', timeout=15000)
            content = await page.content()
            soup    = BeautifulSoup(content, 'html.parser')
            titulo  = soup.find('h1')
            if titulo is None:
                raise ValueError(f"No se encontró h1 para {provincia}")
            span   = titulo.find('span')
            texto  = (span if span else titulo).get_text(strip=True)
            match  = re.search(r"[\d.,]+", texto)
            if not match:
                raise ValueError(f"No se encontró número en: {texto!r}")
            numero = int(match.group().replace(".", "").replace(",", ""))
            resultado[provincia] = numero
            progreso[0] = i + 1

        await browser.close()


def _scrape_en_hilo(urls_provincias: dict, resultado: dict, errores: list, progreso: list):
    """
    Lanza el scraping asíncrono en un hilo separado con su propio event loop.
    Necesario porque Streamlit ya tiene su propio loop y no permite anidarlos.
    """
    import asyncio
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_scrape(urls_provincias, resultado, progreso))
    except Exception as e:
        errores.append(e)
    finally:
        loop.close()


def scrape_n_alojamientos(urls_provincias: dict, barra) -> dict:
    """
    Orquesta el scraping en un hilo daemon y actualiza la barra de progreso
    de Streamlit mientras el hilo trabaja en segundo plano.
    Devuelve un dict {provincia: número de alojamientos}.
    """
    resultado, errores, progreso = {}, [], [0]
    provincias_lista = list(urls_provincias.items())
    total = len(provincias_lista)

    # Lanzamos el scraping en un hilo separado para no bloquear Streamlit
    t = threading.Thread(target=_scrape_en_hilo, args=(urls_provincias, resultado, errores, progreso), daemon=True)
    t.start()

    # Actualizamos la barra de progreso cada 0.5 s mientras el hilo sigue vivo
    while t.is_alive():
        completadas = progreso[0]
        nombre = provincias_lista[completadas][0] if completadas < total else provincias_lista[-1][0]
        barra.progress(completadas / total, text=f'Buscando {nombre}...')
        time.sleep(0.5)

    t.join()

    if errores:
        raise errores[0]

    barra.progress(1.0, text='¡Listo!')
    return resultado



# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    st.set_page_config(page_title='Home')
    st.title('Mapa de Alojamientos de Andalucía')

    col1, col2 = st.columns(2)
    with col1:
        fecha_entrada = st.date_input('Fecha de entrada', min_value=date.today())
    with col2:
        fecha_salida = st.date_input(
            'Fecha de salida',
            value=fecha_entrada + timedelta(days=1),
            min_value=fecha_entrada + timedelta(days=1)
        )

    if 'mapa_listo' not in st.session_state:
        st.session_state.mapa_listo = False

    actualizar = st.button('🔄 Actualizar mapa')

    if actualizar:
        fe_str, fs_str = str(fecha_entrada), str(fecha_salida)
        # Generamos las URLs de Booking para las 8 provincias con las fechas elegidas
        _, urls_provincias = generador_urls(fe_str, fs_str, N_ADULTOS, N_HABITACIONES, N_MENORES, {}, PROVINCIAS)
        # Scrapeamos el número de alojamientos disponibles en cada provincia
        barra        = st.progress(0, text='Iniciando scraping...')
        alojamientos = scrape_n_alojamientos(urls_provincias, barra)
        barra.empty()
        # Generamos el mapa coroplético con los conteos y lo guardamos como HTML
        geojson   = cargar_geojson()
        df_puntos = cargar_puntos_unicos()
        fig       = generar_mapa(alojamientos, geojson, df_puntos, fe_str, fs_str)
        fig.write_html(str(MAPA_ACTUALIZADO))
        st.session_state.mapa_listo = True

    # Mostramos el mapa actualizado si ya se generó, o el predeterminado en caso contrario
    mapa = MAPA_ACTUALIZADO if st.session_state.mapa_listo else MAPA_PREDETERMINADO
    html = mapa.read_text(encoding='utf-8')
    st.iframe(html, height=620)

    mostrar_graficos_analisis()


if __name__ == '__main__':
    main()
