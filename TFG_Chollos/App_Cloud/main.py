'''
Home - Mapa de Alojamientos de Andalucía
=========================================
Para ejecutar:
    python App/run.py
'''

# =============================================================================
# IMPORTS
# =============================================================================
import asyncio
import re
import subprocess
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import streamlit.components.v1 as components

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
    df = pd.read_parquet(
        BASE / 'data' / 'processed' / 'analisis' / 'db_final_analisis.parquet',
        columns=['titulo', 'latitud', 'longitud', 'url_estancia']
    )
    df = df.drop_duplicates(subset='url_estancia')[['titulo', 'latitud', 'longitud']]
    df = df[(df['latitud'].between(35.8, 38.7)) & (df['longitud'].between(-7.6, -1.6))]
    df['_lat_r'] = df['latitud'].round(2)
    df['_lon_r'] = df['longitud'].round(2)
    df = df.drop_duplicates(subset=['_lat_r', '_lon_r'])
    return df[['titulo', 'latitud', 'longitud']]


@st.cache_data
def cargar_mapa_predeterminado():
    if MAPA_PREDETERMINADO.exists():
        return MAPA_PREDETERMINADO.read_text(encoding='utf-8')
    geojson   = cargar_geojson()
    df_puntos = cargar_puntos_unicos()
    fig = generar_mapa({p: 0 for p in NOMBRES_PROVINCIAS}, geojson, df_puntos, 'histórico', '')
    return fig.to_html(include_plotlyjs=True, full_html=True)


def _ensure_playwright_chromium():
    import os
    cache = os.path.expanduser('~/.cache/ms-playwright')
    already = os.path.isdir(cache) and any(
        e.startswith('chromium') for e in os.listdir(cache)
    ) if os.path.isdir(cache) else False
    if not already:
        subprocess.run(
            [sys.executable, '-m', 'playwright', 'install', 'chromium'],
            check=False, timeout=180
        )


async def _async_scrape(urls_provincias: dict, resultado: dict, progreso: list):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-setuid-sandbox",
            ]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            java_script_enabled=True,
            locale="es-ES",
            timezone_id="Europe/Madrid",
        )
        page = await context.new_page()

        async def _abort(route):
            await route.abort()
        await page.route("**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf}", _abort)
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        provincias_lista = list(urls_provincias.items())
        for i, (provincia, url) in enumerate(provincias_lista):
            await page.goto(url, wait_until="load", timeout=30000)
            # Dismiss cookie consent modal if present
            for sel in ['#onetrust-accept-btn-handler', '[data-testid="accept-button"]',
                        'button[aria-label*="ookie"]', 'button[id*="accept"]']:
                try:
                    await page.click(sel, timeout=3000)
                    await asyncio.sleep(0.8)
                    break
                except Exception:
                    pass
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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_scrape(urls_provincias, resultado, progreso))
    except Exception as e:
        errores.append(e)
    finally:
        loop.close()


def scrape_n_alojamientos(urls_provincias: dict, barra) -> dict:
    _ensure_playwright_chromium()
    resultado, errores, progreso = {}, [], [0]
    provincias_lista = list(urls_provincias.items())
    total = len(provincias_lista)

    t = threading.Thread(target=_scrape_en_hilo, args=(urls_provincias, resultado, errores, progreso), daemon=True)
    t.start()

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

    actualizar = st.button('🔄 Actualizar mapa')

    if actualizar:
        fe_str, fs_str = str(fecha_entrada), str(fecha_salida)
        _, urls_provincias = generador_urls(fe_str, fs_str, N_ADULTOS, N_HABITACIONES, N_MENORES, {}, PROVINCIAS)
        barra        = st.progress(0, text='Iniciando scraping...')
        alojamientos = scrape_n_alojamientos(urls_provincias, barra)
        barra.empty()
        geojson   = cargar_geojson()
        df_puntos = cargar_puntos_unicos()
        fig = generar_mapa(alojamientos, geojson, df_puntos, fe_str, fs_str)
        st.session_state['mapa_html'] = fig.to_html(include_plotlyjs=True, full_html=True)

    html = st.session_state.get('mapa_html') or cargar_mapa_predeterminado()
    components.html(html, height=620)

    mostrar_graficos_analisis()


if __name__ == '__main__':
    main()
