'''
_scraper_app.py
===============
Helper de scraping para la app Streamlit. No es una página Streamlit.
Exporta:
    scrape_busqueda(urls, fecha_entrada, fecha_salida, barra)
    → list[dict] de datos crudos para _predictor.preprocesar_nuevos()

Flujo:
    1. Playwright (async via hilo) → listado Booking → info básica + URLs
    2. BookingExtractor (ThreadPoolExecutor) → ficha de detalle por alojamiento
'''

# =============================================================================
# IMPORTS
# =============================================================================
import asyncio
import functools
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from urllib.parse import parse_qs, quote, urlencode, urlsplit

import requests as _req
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

from TFG_Chollos.Scraping.Scrp_caracteristicas_estancias import (
    BOOKING_INTERNAL_IDS,
    CALENDAR_QUERY,
    GRAPHQL_URL,
    HOTEL_ID_PATTERNS,
    BookingExtractor,
)

# =============================================================================
# CONSTANTES
# =============================================================================
N_ADULTOS      = 2
N_HABITACIONES = 1
N_MENORES      = 0
MAX_CARDS      = 25

# Caché en memoria de (ss, dest_id, dest_type) resueltos vía autocomplete,
# para no repetir la navegación a la home en cada búsqueda del mismo lugar
# dentro de la misma sesión de Streamlit.
_DEST_CACHE: dict[str, tuple[str, str, str]] = {}

# Coordenadas hardcodeadas de municipios andaluces (fuente: IGN / OSM).
# Evita dependencia de servicios externos de geocoding (Nominatim/Photon)
# que pueden estar bloqueados desde Streamlit Cloud.
_COORDS_ANDALUCIA: dict[str, tuple[float, float]] = {
    # ── Capitales de provincia ─────────────────────────────────────────────
    'Sevilla':    (37.3891, -5.9845), 'Málaga':   (36.7213, -4.4213),
    'Córdoba':   (37.8882, -4.7794), 'Granada':  (37.1773, -3.5986),
    'Almería':   (36.8340, -2.4637), 'Jaén':     (37.7796, -3.7849),
    'Cádiz':     (36.5271, -6.2886), 'Huelva':   (37.2614, -6.9447),
    # ── Costa del Sol (Málaga) ─────────────────────────────────────────────
    'Marbella':  (36.5101, -4.8825), 'Torremolinos': (36.6217, -4.4994),
    'Fuengirola': (36.5401, -4.6250), 'Estepona': (36.4272, -5.1461),
    'Benalmádena': (36.5983, -4.5207), 'Nerja':   (36.7450, -3.8793),
    'Mijas':     (36.5990, -4.6361), 'Vélez-Málaga': (36.7810, -4.0985),
    'Rincón de la Victoria': (36.7178, -4.2793),
    'Alhaurín de la Torre': (36.6640, -4.5578),
    'Antequera': (37.0200, -4.5620), 'Ronda':    (36.7459, -5.1616),
    # ── Costa de la Luz — Cádiz ────────────────────────────────────────────
    'Rota':      (36.6248, -6.3615), 'Chipiona': (36.7349, -6.4374),
    'Sanlúcar de Barrameda': (36.7759, -6.3536),
    'El Puerto de Santa María': (36.5930, -6.2330),
    'Chiclana de la Frontera': (36.4173, -6.1477),
    'Conil de la Frontera': (36.2775, -6.0879),
    'Vejer de la Frontera': (36.2520, -5.9699),
    'Barbate':   (36.1906, -5.9215), 'Tarifa':   (36.0143, -5.6051),
    'Algeciras': (36.1327, -5.4541),
    'La Línea de la Concepción': (36.1672, -5.3494),
    'San Fernando': (36.4778, -6.1986), 'Puerto Real': (36.5278, -6.1938),
    'Jerez de la Frontera': (36.6866, -6.1370),
    'San Roque': (36.2139, -5.3885), 'Los Barrios': (36.1888, -5.4948),
    # ── Costa de la Luz — Huelva ───────────────────────────────────────────
    'Punta Umbría': (37.1747, -6.9670), 'Ayamonte': (37.2140, -7.4040),
    'Isla Cristina': (37.1990, -7.3270), 'Lepe':   (37.2534, -7.2047),
    'Cartaya':   (37.2773, -7.1441), 'Palos de la Frontera': (37.2290, -6.8938),
    # ── Costa de Granada y Almería ─────────────────────────────────────────
    'Almuñécar': (36.7319, -3.6936), 'Motril':   (36.7511, -3.5215),
    'Salobreña': (36.7416, -3.5877), 'Adra':     (36.7483, -3.0247),
    'Roquetas de Mar': (36.7642, -2.6149), 'El Ejido': (36.7768, -2.8119),
    'Vera':      (37.2503, -1.8627), 'Mojácar':  (37.1418, -1.8447),
    'Carboneras': (37.0010, -1.8939),
    # ── Interior — Sevilla ────────────────────────────────────────────────
    'Dos Hermanas': (37.2810, -5.9219), 'Alcalá de Guadaíra': (37.3355, -5.8467),
    'Utrera':    (37.1847, -5.7800), 'Écija':    (37.5432, -5.0823),
    'Carmona':   (37.4715, -5.6447), 'Osuna':    (37.2360, -5.1079),
    'Lebrija':   (36.9189, -6.0746), 'Morón de la Frontera': (37.1233, -5.4508),
    'Marchena':  (37.3311, -5.3813), 'Estepa':   (37.2887, -4.8777),
    # ── Interior — Córdoba ───────────────────────────────────────────────
    'Lucena':    (37.4091, -4.4852), 'Montilla': (37.5849, -4.6408),
    'Cabra':     (37.4710, -4.4420), 'Puente Genil': (37.3916, -4.7661),
    'Priego de Córdoba': (37.4371, -4.1960), 'Palma del Río': (37.7009, -5.2807),
    # ── Interior — Jaén ──────────────────────────────────────────────────
    'Linares':   (38.0950, -3.6355), 'Andújar':  (38.0394, -4.0504),
    'Úbeda':     (38.0132, -3.3706), 'Baeza':    (37.9919, -3.4706),
    'Martos':    (37.7227, -3.9668), 'Alcalá la Real': (37.4601, -3.9239),
    'Cazorla':   (37.9154, -2.9454), 'Villacarrillo': (38.1153, -3.0809),
    # ── Interior — Granada ───────────────────────────────────────────────
    'Guadix':    (37.2992, -3.1391), 'Baza':     (37.4935, -2.7699),
    'Loja':      (37.1694, -4.1503), 'Huéscar':  (37.8119, -2.5415),
    # ── Interior — Almería ───────────────────────────────────────────────
    'Roquetas de Mar': (36.7642, -2.6149), 'Níjar': (36.9660, -2.2026),
}


def _ensure_playwright_chromium(barra=None):
    """Instala el Chromium propio de Playwright si no está disponible."""
    import os
    cache = os.path.expanduser('~/.cache/ms-playwright')
    already = os.path.isdir(cache) and any(
        e.startswith('chromium') for e in os.listdir(cache)
    ) if os.path.isdir(cache) else False
    if not already:
        if barra:
            barra.progress(0, text='Instalando Chromium de Playwright...')
        subprocess.run(
            [sys.executable, '-m', 'playwright', 'install', 'chromium'],
            check=False, timeout=180
        )


@functools.lru_cache(maxsize=128)
def _resolver_coords(lugar: str) -> tuple[float, float, str]:
    """
    Obtiene lat/lon para un municipio andaluz. Resultado cacheado en memoria.
    Estrategia (de más a menos fiable):
      1. Dict hardcodeado _COORDS_ANDALUCIA — sin red, instantáneo
      2. Photon (photon.komoot.io)           — OSM, sin rate-limit estricto
      3. Nominatim (fallback)                — OSM, gestiona 429
    """
    headers = {'User-Agent': 'TFGChollos/1.0 (mariosevillista002@gmail.com)'}

    # ── 1. Dict local (sin red) ────────────────────────────────────────────────
    lat, lon = _COORDS_ANDALUCIA.get(lugar, (0.0, 0.0))
    if lat != 0.0:
        return lat, lon, f'lat={lat:.4f} lon={lon:.4f} [dict]'

    # ── 2. Photon ──────────────────────────────────────────────────────────────
    for q in [f'{lugar}, Andalucía, España', f'{lugar}, España', lugar]:
        try:
            resp = _req.get(
                'https://photon.komoot.io/api/',
                params={'q': q, 'limit': 1, 'lang': 'es'},
                headers=headers,
                timeout=10,
            )
            if resp.ok:
                feats = resp.json().get('features', [])
                if feats:
                    lon_f, lat_f = feats[0]['geometry']['coordinates']
                    if 36.0 <= lat_f <= 44.0 and -9.5 <= lon_f <= 4.5:
                        return float(lat_f), float(lon_f), f'lat={lat_f:.4f} lon={lon_f:.4f} [Photon]'
        except Exception:
            pass

    # ── 3. Nominatim (fallback) ────────────────────────────────────────────────
    nominatim_queries = [
        {'q': f'{lugar}, Andalucía', 'format': 'json', 'limit': '1', 'countrycodes': 'es'},
        {'q': lugar,                 'format': 'json', 'limit': '1', 'countrycodes': 'es'},
        {'q': f'{lugar}, España',    'format': 'json', 'limit': '1'},
    ]
    last_err = 'sin resultados en dict/Photon/Nominatim'
    for i, params in enumerate(nominatim_queries):
        if i > 0:
            time.sleep(1)
        for attempt in range(2):
            try:
                resp = _req.get(
                    'https://nominatim.openstreetmap.org/search',
                    params=params, headers=headers, timeout=10,
                )
                if resp.ok:
                    data = resp.json()
                    if data:
                        lat = float(data[0]['lat'])
                        lon = float(data[0]['lon'])
                        return lat, lon, f'lat={lat:.4f} lon={lon:.4f} [Nominatim]'
                    last_err = f'sin resultados Nominatim q={params["q"]!r}'
                    break
                elif resp.status_code == 429:
                    wait = int(resp.headers.get('Retry-After', 3))
                    last_err = f'429 Nominatim'
                    time.sleep(max(wait, 1))
                else:
                    last_err = f'HTTP {resp.status_code} Nominatim'
                    break
            except Exception as _e:
                last_err = f'{type(_e).__name__} Nominatim'
                if attempt == 0:
                    time.sleep(1)

    return 0.0, 0.0, f'geocoding falló: {last_err}'


def _url_por_coords(lat: float, lon: float, fecha_entrada: str, fecha_salida: str) -> str:
    return (
        'https://www.booking.com/searchresults.es.html'
        f'?latitude={lat}'
        f'&longitude={lon}'
        f'&radius=10'
        f'&checkin={fecha_entrada}'
        f'&checkout={fecha_salida}'
        f'&group_adults={N_ADULTOS}'
        f'&no_rooms={N_HABITACIONES}'
        f'&group_children={N_MENORES}'
        f'&selected_currency=EUR'
    )


# =============================================================================
# FASE 2 — FIX LOCAL: servicios/lat/lon/dirección vía Playwright
# =============================================================================
def _fetch_location_y_servicios(url: str) -> dict:
    """
    Copia local (con fix) de BookingExtractor._location()/_amenities()
    (TFG_Chollos/Scraping/Scrp_caracteristicas_estancias.py).

    process_row() obtiene el HTML de la ficha con self.http.get(url) (requests),
    que en Streamlit Cloud devuelve una página degradada sin JSON-LD ni el
    wrapper de servicios más populares, dejando "servicios", "latitud" y
    "longitud" vacíos. _room_amenities() sí funciona porque carga la página
    con Playwright (ejecuta JS); replicamos ese mismo enfoque aquí para
    obtener el HTML completo y extraer de él lat/lon, dirección y servicios.

    NOTA: se probó a reutilizar un navegador por hilo en vez de lanzar uno
    nuevo por ficha (para reducir el tiempo de scraping en Streamlit Cloud),
    pero en producción (donde este fallback se ejecuta para casi todas las
    fichas, al contrario que en local) provocaba que la mayoría de fichas se
    perdieran silenciosamente. Se revierte a lanzar y cerrar un navegador
    nuevo por llamada, que es el comportamiento ya probado y estable.
    """
    extra = {
        'latitud': '', 'longitud': '', 'direccion': '', 'ciudad': '',
        'pais': '', 'codigo_postal': '', 'servicios': [], '_html': '',
    }
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-setuid-sandbox',
                ],
            )
            context = browser.new_context(
                user_agent=(
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
                ),
                locale='es-ES',
            )
            context.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
                "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3]});"
            )
            page = context.new_page()
            page.route(
                '**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,css}',
                lambda route: route.abort(),
            )
            # JSON-LD (geo/dirección) y el wrapper de servicios ya vienen en el
            # HTML servido inicialmente, no requieren esperar a 'networkidle'
            # (que en Booking nunca termina de verdad por los pings de analítica).
            try:
                page.goto(url, wait_until='domcontentloaded', timeout=20_000)
            except Exception:
                pass
            try:
                # 3s en vez de 5s: solo se nota en el peor caso (la página no
                # trae JSON-LD), donde de todos modos se sigue sin ese dato.
                page.wait_for_selector('script[type="application/ld+json"]', timeout=3_000)
            except Exception:
                pass
            html = page.content()
            browser.close()
    except Exception:
        return extra

    extra['_html'] = html
    soup = BeautifulSoup(html, 'html.parser')

    # ── Dirección y coordenadas: JSON-LD (schema.org) ───────────────────────
    for s in soup.find_all('script', {'type': 'application/ld+json'}):
        try:
            obj  = json.loads(s.string or '')
            geo  = obj.get('geo') or {}
            addr = obj.get('address') or {}
            if geo or addr:
                extra.update({
                    'latitud':       str(geo.get('latitude',  '')),
                    'longitud':      str(geo.get('longitude', '')),
                    'direccion':     addr.get('streetAddress', ''),
                    'ciudad':        addr.get('addressLocality', ''),
                    'pais':          addr.get('addressCountry', ''),
                    'codigo_postal': addr.get('postalCode', ''),
                })
        except Exception:
            pass

    # ── Fallback de coordenadas vía regex sobre el HTML crudo ───────────────
    if not extra['latitud']:
        for p in [r'"latitude"\s*:\s*([-\d.]+)', r'b_map_center_lat\s*=\s*([-\d.]+)']:
            m = re.search(p, html)
            if m:
                extra['latitud'] = m.group(1)
                break
    if not extra['longitud']:
        for p in [r'"longitude"\s*:\s*([-\d.]+)', r'b_map_center_lon\s*=\s*([-\d.]+)']:
            m = re.search(p, html)
            if m:
                extra['longitud'] = m.group(1)
                break

    # ── Servicios generales: wrapper de "servicios más populares" ───────────
    found = set()
    for wrapper in soup.select('[data-testid="property-most-popular-facilities-wrapper"]'):
        for li in wrapper.select('li'):
            icon = li.select_one('[data-testid="facility-icon"]')
            if icon and icon.parent:
                for child in icon.parent.children:
                    if child == icon:
                        continue
                    if hasattr(child, 'get_text'):
                        t = child.get_text(strip=True)
                        if t and len(t) > 2:
                            found.add(t)
                            break
    extra['servicios'] = sorted(found)

    return extra


def _extraer_hotel_id(html_text: str, extractor: BookingExtractor) -> str:
    """
    Copia local (con fix) de BookingExtractor._get_hotel_id(), limitada a las
    estrategias 1 y 2 (regex sobre el HTML + __NEXT_DATA__).

    process_row() obtiene el hotel_id a partir del HTML de self.http.get(url)
    (requests), que en Streamlit Cloud devuelve una página degradada sin
    __NEXT_DATA__ ni los campos que buscan los patrones, dejando "hotel_id"
    vacío y el calendario sin consultar. Reutilizamos el HTML completo ya
    obtenido vía Playwright en _fetch_location_y_servicios.
    """
    for pattern in HOTEL_ID_PATTERNS:
        m = re.search(pattern, html_text)
        if m and len(m.group(1)) >= 4 and m.group(1) not in BOOKING_INTERNAL_IDS:
            return m.group(1)

    return extractor._get_hotel_id_from_next_data(html_text) or ''


def _debug_calendar_query(extractor: BookingExtractor, hotel_id: str, pagename: str,
                           country_code: str, checkin: str, adults: int) -> str:
    """
    Réplica de un único chunk de BookingExtractor._calendar(), solo para
    diagnóstico: hace la misma petición GraphQL que _calendar() y devuelve
    una descripción de la respuesta cruda (o el motivo por el que
    post_json() no devolvió nada), para averiguar si dias_calendario=0 se
    debe a un bloqueo WAF en self.http o a que Booking no devuelve días.
    """
    referer = f"https://www.booking.com/hotel/{country_code}/{pagename}.es.html"
    payload = {
        "operationName": "AvailabilityCalendar",
        "query": CALENDAR_QUERY,
        "variables": {
            "input": {
                "travelPurpose": 2,
                "pagenameDetails": {"countryCode": country_code, "pagename": pagename},
                "searchConfig": {
                    "searchConfigDate": {"startDate": checkin, "amountOfDays": 90},
                    "nbAdults": adults, "nbChildren": 0,
                    "nbRooms": 1, "childrenAges": [],
                },
            }
        },
    }
    data = extractor.http.post_json(GRAPHQL_URL, payload, referer=referer)
    if data is None:
        return 'post_json devolvió None (sin respuesta válida / WAF)'
    dias = data.get('data', {}).get('availabilityCalendar', {}).get('days', [])
    return f'post_json OK | días={len(dias)} | data={json.dumps(data, ensure_ascii=False)[:400]}'


# =============================================================================
# FASE 1 — SCRAPING DEL LISTADO CON PLAYWRIGHT
# =============================================================================
async def _async_scrape_listado(lugar: str, url: str, fecha_entrada: str,
                                 fecha_salida: str, resultado: list, errores: list,
                                 debug_info: list):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-setuid-sandbox',
                ]
            )

            # ── PASO 1: scraping del listado de resultados ─────────────────────────
            # Usamos primero la URL ss=... (búsqueda por nombre de lugar) generada
            # en Busqueda.py, que ya incluye los filtros de tipo/servicios (nflt=)
            # elegidos por el usuario y da el conteo "Hemos encontrado N alojamientos"
            # que el usuario ve al buscar manualmente. La búsqueda por coordenadas
            # (radius=10km) es mucho más amplia (cientos de resultados) y solo se usa
            # como último recurso si Booking devuelve la homepage (bot detection).
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                java_script_enabled=True,
                locale='es-ES',
                timezone_id='Europe/Madrid',
            )
            page = await context.new_page()

            async def _abort(route):
                await route.abort()

            await page.route('**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf}', _abort)
            await page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

            _cookies_dismissed = False

            async def _dismiss_cookies():
                # El banner de cookies solo aparece (como mucho) en la primera
                # navegación de la sesión, así que no repetimos la búsqueda en
                # cada page.goto posterior.
                nonlocal _cookies_dismissed
                if _cookies_dismissed:
                    return
                _cookies_dismissed = True
                for sel in ['#onetrust-accept-btn-handler', '[data-testid="accept-button"]',
                            'button[aria-label*="ookie"]', 'button[id*="accept"]']:
                    try:
                        await page.click(sel, timeout=1500)
                        await asyncio.sleep(0.5)
                        break
                    except Exception:
                        pass

            async def _load_and_capture(target_url: str, etiqueta: str = 'principal') -> str:
                await page.goto(target_url, wait_until='domcontentloaded', timeout=30000)
                await _dismiss_cookies()
                try:
                    await page.wait_for_selector('[data-testid="property-card"]', timeout=15000)
                except Exception:
                    pass

                # El <h1 aria-live="assertive"> muestra inicialmente el total SIN
                # filtrar (coincide con su aria-label, p.ej. "739 alojamientos
                # encontrados") y solo pasa al conteo FILTRADO por fechas (p.ej.
                # "15 alojamientos encontrados") cuando React recibe la respuesta
                # del filtro. Esperamos a que el texto del h1 cambie respecto a su
                # valor inicial (hasta ~8s) para no quedarnos con el total sin filtrar.
                h1 = page.locator('h1[aria-live="assertive"]').first
                _h1_count = await page.locator('h1[aria-live="assertive"]').count()
                try:
                    _texto_inicial = await h1.inner_text(timeout=3000)
                except Exception:
                    _texto_inicial = None

                # Techo de espera reducido de 8s a 4s (10 x 0.4s): el bucle ya
                # corta en cuanto detecta el cambio, esto solo baja el peor caso.
                _texto_final = _texto_inicial
                if _texto_inicial is not None:
                    for _ in range(10):
                        await asyncio.sleep(0.4)
                        try:
                            _texto_actual = await h1.inner_text(timeout=1000)
                        except Exception:
                            break
                        _texto_final = _texto_actual
                        if _texto_actual != _texto_inicial:
                            break
                else:
                    try:
                        await page.wait_for_load_state('networkidle', timeout=6000)
                    except Exception:
                        pass

                debug_info.append(
                    f'[H1 DEBUG {etiqueta}] count={_h1_count} '
                    f'inicial={_texto_inicial!r} final={_texto_final!r}'
                )

                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                await asyncio.sleep(1.0)

                _html = await page.content()
                return _html

            async def _resolver_dest_via_autocomplete() -> str | None:
                """
                Una URL searchresults.es.html?ss=Lugar sin dest_id/dest_type no es
                reconocida por Booking y redirige a la homepage. Para conseguir un
                dest_id/dest_type válidos (y por tanto el mismo área/conteo que ve
                un usuario real), pasamos por la home, escribimos el destino en el
                buscador y usamos la sugerencia del autocompletado.

                El resultado (ss, dest_id, dest_type) se cachea por lugar para no
                repetir esta navegación extra en cada búsqueda. No cacheamos fallos:
                pueden ser transitorios (timeout, WAF puntual) y cachear None dejaría
                ese lugar roto (cayendo siempre al fallback de coordenadas) durante
                toda la vida del proceso.
                """
                def _build_url(ss: str, dest_id: str, dest_type: str) -> str:
                    params = {
                        'ss':             ss,
                        'dest_id':        dest_id,
                        'dest_type':      dest_type,
                        'checkin':        fecha_entrada,
                        'checkout':       fecha_salida,
                        'group_adults':   N_ADULTOS,
                        'no_rooms':       N_HABITACIONES,
                        'group_children': N_MENORES,
                        'selected_currency': 'EUR',
                    }
                    return 'https://www.booking.com/searchresults.es.html?' + urlencode(params)

                if lugar in _DEST_CACHE:
                    return _build_url(*_DEST_CACHE[lugar])

                try:
                    await page.goto('https://www.booking.com/index.es.html', wait_until='domcontentloaded', timeout=30000)
                    await _dismiss_cookies()

                    ss_input = page.locator('input[name="ss"]')
                    await ss_input.click(timeout=8000)
                    await ss_input.fill('')
                    await ss_input.type(lugar, delay=50)

                    sugerencia = page.locator('[data-testid="autocomplete-result"]').first
                    await sugerencia.wait_for(timeout=6000)
                    await sugerencia.click()

                    await page.locator('button[type="submit"]').first.click()
                    await page.wait_for_url('**/searchresults*', timeout=15000)

                    _qs = parse_qs(urlsplit(page.url).query)
                    dest_id   = _qs.get('dest_id', [None])[0]
                    dest_type = _qs.get('dest_type', [None])[0]
                    ss        = _qs.get('ss', [lugar])[0]
                    if not dest_id or not dest_type:
                        return None

                    _DEST_CACHE[lugar] = (ss, dest_id, dest_type)
                    return _build_url(ss, dest_id, dest_type)
                except Exception:
                    return None

            dest_id_debug = 'ss= directo'
            _ui_url = await _resolver_dest_via_autocomplete()
            if _ui_url:
                _nflt = parse_qs(urlsplit(url).query).get('nflt', [None])[0]
                if _nflt:
                    _ui_url += '&nflt=' + quote(_nflt, safe='')
                url = _ui_url
                dest_id_debug = 'autocomplete UI'

            content = await _load_and_capture(url)
            soup = BeautifulSoup(content, 'html.parser')

            page_title = soup.find('title')
            title_text = page_title.get_text(strip=True)[:60] if page_title else '(sin título)'

            # Detectar si Booking devolvió la homepage (bot detection) en vez de resultados
            _es_homepage = 'sitio oficial' in title_text.lower() or 'booking.com |' in title_text.lower()

            debug_info.append(
                f'[PASO1 DEBUG] dest_id_debug={dest_id_debug} | es_homepage={_es_homepage} | '
                f'título={title_text!r} | tarjetas={len(soup.find_all("div", {"data-testid": "property-card"}))} | '
                f'url={url}'
            )

            # ── PASO 2 (fallback): si nos detectaron como bot, reintentar con una
            # búsqueda por coordenadas, conservando el filtro nflt= de la URL original ──
            if _es_homepage:
                try:
                    lat, lon, dest_id_debug = _resolver_coords(lugar)
                    if lat != 0.0:
                        _nflt = parse_qs(urlsplit(url).query).get('nflt', [None])[0]
                        url = _url_por_coords(lat, lon, fecha_entrada, fecha_salida)
                        if _nflt:
                            url += '&nflt=' + quote(_nflt, safe='')
                        content = await _load_and_capture(url, etiqueta='fallback-coords')
                        soup = BeautifulSoup(content, 'html.parser')
                        page_title = soup.find('title')
                        title_text = page_title.get_text(strip=True)[:60] if page_title else '(sin título)'
                        _es_homepage = 'sitio oficial' in title_text.lower() or 'booking.com |' in title_text.lower()
                    else:
                        dest_id_debug = f'homepage detectada, sin coords: {dest_id_debug}'
                except Exception as _e:
                    dest_id_debug = f'FALLO fallback coords: {type(_e).__name__}: {str(_e)[:80]}'

            await browser.close()

        n_reales = 0
        if not _es_homepage:
            # Patrones de mayor a menor especificidad: primero el formato del h1
            # de resultados ya filtrados por fecha, p.ej. "Rota: 15 alojamientos
            # encontrados" (el número va ANTES de "encontrado/s"); luego "Hemos
            # encontrado X alojamientos"; y por último cualquier "X alojamiento/
            # propiedad/resultado" suelto (puede venir de texto SEO con el total
            # SIN filtrar, p.ej. "725 alojamientos").
            _patterns = [
                r'(\d[\d.]*)\s+(?:alojamiento|propiedad|resultado)\w*\s+(?:encontrado|encontramos)',
                r'(?:encontrado|encontramos)\s+(\d[\d.]*)\s+(?:alojamiento|propiedad|resultado)',
                r'(\d[\d.]*)\s+(?:alojamiento|propiedad|resultado)',
            ]
            # 1. Intentar parsear el conteo real desde el h1
            for _sel in [
                soup.find('h1', {'aria-live': 'assertive'}),
                soup.find('h1'),
            ]:
                if _sel:
                    _texto = _sel.get_text(separator=' ', strip=True)
                    for _pat in _patterns:
                        _m = re.search(_pat, _texto, re.IGNORECASE)
                        if _m:
                            n_reales = int(_m.group(1).replace('.', ''))
                            break
                    if n_reales:
                        break
            # 2. Fallback: buscar el número en todo el texto de la página
            if n_reales == 0:
                _full_text = soup.get_text(' ', strip=True)
                for _pat in _patterns:
                    _m = re.search(_pat, _full_text, re.IGNORECASE)
                    if _m:
                        _candidate = int(_m.group(1).replace('.', ''))
                        if 1 <= _candidate <= 500:   # sanity check
                            n_reales = _candidate
                            break

        estancias = soup.find_all('div', {'data-testid': 'property-card'})
        if n_reales == 0:
            n_reales = len(estancias)

        debug_info.append(f'Título: {title_text!r} | Tarjetas: {len(estancias)} | n_reales: {n_reales} | Autocomplete: {dest_id_debug} | URL: {url}')

        _precio_debug_pendiente = 2  # nº de tarjetas de las que volcamos el HTML del precio

        for estancia in estancias[:min(n_reales, MAX_CARDS)]:
            try:
                enlace = estancia.find('a', {'data-testid': 'title-link'})
                url_base = enlace['href'].split('?')[0]
                url_estancia = (
                    f'{url_base}'
                    f'?checkin={fecha_entrada}'
                    f'&checkout={fecha_salida}'
                    f'&group_adults={N_ADULTOS}'
                    f'&req_adults={N_ADULTOS}'
                    f'&no_rooms={N_HABITACIONES}'
                    f'&group_children={N_MENORES}'
                    f'&req_children={N_MENORES}'
                    f'&selected_currency=EUR'
                )

                nombre = estancia.find('div', {'data-testid': 'title'})
                titulo = nombre.get_text(strip=True)

                # Precio mostrado en la tarjeta de búsqueda (precio total de la estancia
                # con las fechas seleccionadas). El testid "price-and-discounted-price"
                # puede incluir el precio original tachado seguido del precio con
                # descuento (p.ej. "€ 400 € 332"), por lo que nos quedamos con el
                # último número, que es siempre el precio final a pagar.
                precio_elem = estancia.find('span', {'data-testid': 'price-and-discounted-price'})
                if precio_elem:
                    _precios = re.findall(r'\d[\d.,]*', precio_elem.get_text(' ', strip=True))
                    if _precios:
                        _ultimo = re.sub(r'[^\d]', '', _precios[-1])
                        precio_listado = int(_ultimo) if _ultimo else None
                    else:
                        precio_listado = None
                else:
                    precio_listado = None

                if _precio_debug_pendiente > 0:
                    _precio_debug_pendiente -= 1
                    _html_precio = str(precio_elem)[:500] if precio_elem else '(sin price-and-discounted-price)'
                    debug_info.append(
                        f'[PRECIO DEBUG] título={titulo!r} | precio_listado={precio_listado} | '
                        f'_precios={_precios if precio_elem else None} | html={_html_precio}'
                    )

                # Precio total de la estancia: primero data-testid conocidos de Booking,
                # luego búsqueda de texto "X € total" / "€ X en total" en toda la tarjeta.
                precio_total_card = None
                for _tid in [
                    re.compile(r'price-for-\d+-nights?', re.I),
                    re.compile(r'total-price|inclusive-price|taxes-and-charges', re.I),
                ]:
                    _e = estancia.find(attrs={'data-testid': _tid})
                    if _e:
                        _n = re.sub(r'[^\d]', '', _e.get_text(strip=True))
                        _v = int(_n) if _n else None
                        if _v and (precio_listado is None or _v >= precio_listado):
                            precio_total_card = _v
                            break
                if precio_total_card is None:
                    _card_text = estancia.get_text(' ', strip=True)
                    for _pat in [
                        r'€\s*([\d][0-9.]*)\s*(?:en total|total)',   # "€ 360 en total"
                        r'([\d][0-9.]*)\s*€\s*(?:en total|total)',   # "360 € total"
                        r'(?:en total|total)\s*:?\s*€?\s*([\d][0-9.]*)',  # "total: 360"
                    ]:
                        _m = re.search(_pat, _card_text, re.IGNORECASE)
                        if _m:
                            _n = re.sub(r'[^\d]', '', _m.group(1))
                            _v = int(_n) if _n else None
                            if _v and (precio_listado is None or _v >= precio_listado):
                                precio_total_card = _v
                                break

                reviews = estancia.find('div', {'data-testid': 'review-score'})
                if reviews:
                    valoracion = re.search(r'\d+(?:[.,]\d)?', reviews.get_text()).group()
                    num_val    = re.search(r'\d[\d.]*\s+comentarios?', reviews.get_text())
                    n_val      = num_val.group().replace('.', '').split(' ')[0] if num_val else '0'
                else:
                    valoracion = 'NA'
                    n_val      = '0'

                if estancia.find('div', {'data-testid': 'rating-squares'}):
                    tipo      = 'Otro'
                    n_est     = estancia.find('div', {'data-testid': 'rating-squares'})
                    estrellas = len(n_est.find_all('div', recursive=False))
                elif estancia.find('div', {'data-testid': 'rating-stars'}):
                    tipo      = 'Hotel'
                    n_est     = estancia.find('div', {'data-testid': 'rating-stars'})
                    estrellas = len(n_est.find_all('div', recursive=False))
                else:
                    tipo      = 'Otro'
                    estrellas = 1

                resultado.append({
                    'lugar':                    lugar,
                    'titulo':                   titulo,
                    'url_estancia':             url_estancia,
                    'precio_listado':           precio_listado,
                    'precio_total_card':        precio_total_card,
                    'valoracion_clientes':      valoracion,
                    'n_valoraciones':           n_val,
                    'tipo':                     tipo,
                    'estrellas':                estrellas,
                    'fecha_entrada':            fecha_entrada,
                    'fecha_salida':             fecha_salida,
                    'n_adultos':                N_ADULTOS,
                    'n_habitaciones':           N_HABITACIONES,
                    'n_menores':                N_MENORES,
                    'fecha_extraccion_listado': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })
            except Exception:
                pass

    except Exception as e:
        errores.append(e)


def _listado_en_hilo(lugar: str, url: str, fecha_entrada: str, fecha_salida: str,
                     resultado: list, errores: list, debug_info: list):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _async_scrape_listado(lugar, url, fecha_entrada, fecha_salida, resultado, errores, debug_info)
        )
    finally:
        loop.close()


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def scrape_busqueda(urls: dict, fecha_entrada: str, fecha_salida: str, barra,
                     diag_out: list | None = None) -> list[dict]:
    '''
    Orquesta el scraping completo para los destinos indicados.

    Parámetros
    ----------
    urls          : {lugar: url_booking} generado por Busqueda.py
    fecha_entrada : str "YYYY-MM-DD"
    fecha_salida  : str "YYYY-MM-DD"
    barra         : st.progress() para mostrar progreso
    diag_out      : si se pasa una lista, se rellena con el diagnóstico
                    (Tarjetas/n_reales/etc.) de cada destino, para debug

    Devuelve
    --------
    Lista de dicts crudos lista para _predictor.preprocesar_nuevos()
    '''
    _ensure_playwright_chromium(barra)

    destinos = list(urls.items())
    total_destinos = len(destinos)

    # ── FASE 1: listados ─────────────────────────────────────────────────────
    listados = []
    diagnosticos = []
    for i, (lugar, url) in enumerate(destinos):
        barra.progress(
            i / (total_destinos * 2),
            text=f'Buscando alojamientos en {lugar}...'
        )
        resultado, errores, debug_info = [], [], []
        t = threading.Thread(
            target=_listado_en_hilo,
            args=(lugar, url, fecha_entrada, fecha_salida, resultado, errores, debug_info),
            daemon=True,
        )
        t.start()
        t.join()
        if errores:
            raise RuntimeError(f'Error scraping {lugar}: {errores[0]}') from errores[0]
        if debug_info:
            diagnosticos.append(f'{lugar}: {debug_info[0]}')
            for _extra in debug_info[1:]:
                diagnosticos.append(f'  {lugar} | {_extra}')
        else:
            diagnosticos.append(f'{lugar}: sin diagnóstico')
        listados.extend(resultado)

    # Deduplicar por URL base para evitar fichas repetidas (sponsored cards, etc.)
    _seen = set()
    _dedup = []
    for _item in listados:
        _k = _item.get('url_estancia', '').split('?')[0]
        if _k and _k not in _seen:
            _seen.add(_k)
            _dedup.append(_item)
    listados = _dedup

    if diag_out is not None:
        diag_out.extend(diagnosticos)

    if not listados:
        resumen = ' | '.join(diagnosticos) if diagnosticos else 'sin datos'
        barra.progress(1.0, text=f'0 resultados — {resumen}')
        return []

    # ── FASE 2: fichas de detalle ─────────────────────────────────────────────
    barra.progress(0.5, text=f'Obteniendo detalles de {len(listados)} alojamientos...')

    extractor = BookingExtractor()
    fichas    = []
    procesados = [0]

    def _procesar(row, idx):
        try:
            ficha = extractor.process_row(row, idx, len(listados))
        except Exception:
            return None

        if not ficha or ficha.get('error'):
            return ficha

        # self.http.get(url) (requests) suele devolver una página degradada en
        # Streamlit Cloud, sin JSON-LD ni el wrapper de servicios → "servicios",
        # "latitud" y "longitud" llegan vacíos. Si falta algo, lo recuperamos
        # con una carga real vía Playwright (ver _fetch_location_y_servicios).
        falta_servicios = ficha.get('servicios') in (None, '', '[]')
        falta_coords    = not ficha.get('latitud') or not ficha.get('longitud')
        falta_hotel_id  = not ficha.get('hotel_id')
        if falta_servicios or falta_coords or falta_hotel_id:
            extra = _fetch_location_y_servicios(ficha.get('url_estancia', ''))
            if falta_coords and extra['latitud'] and extra['longitud']:
                ficha['latitud']  = extra['latitud']
                ficha['longitud'] = extra['longitud']
                ficha['google_maps'] = (
                    f"https://www.google.com/maps?q={extra['latitud']},{extra['longitud']}"
                )
            for campo in ('direccion', 'ciudad', 'pais', 'codigo_postal'):
                if not ficha.get(campo) and extra[campo]:
                    ficha[campo] = extra[campo]
            if falta_servicios and extra['servicios']:
                ficha['servicios'] = json.dumps(extra['servicios'], ensure_ascii=False)
            if falta_hotel_id and extra['_html']:
                hotel_id = _extraer_hotel_id(extra['_html'], extractor)
                if hotel_id:
                    ficha['hotel_id'] = hotel_id
                    url_estancia = ficha.get('url_estancia', '')
                    pagename     = extractor._pagename(url_estancia)
                    country_code = extractor._country(url_estancia)
                    try:
                        adultos = int(row.get('n_adultos', 2) or 2)
                    except ValueError:
                        adultos = 2
                    try:
                        calendar = extractor._calendar(
                            hotel_id, pagename, country_code,
                            row.get('fecha_entrada', ''), row.get('fecha_salida', ''),
                            adultos,
                        )
                        ficha['calendario'] = json.dumps(calendar, ensure_ascii=False)
                    except Exception:
                        calendar = []

                    if diag_out is not None and idx <= 2 and not calendar:
                        _raw = _debug_calendar_query(
                            extractor, hotel_id, pagename, country_code,
                            row.get('fecha_entrada', ''), adultos,
                        )
                        diag_out.append(
                            f'[CALENDARIO RAW DEBUG] título={ficha.get("titulo","")!r} | {_raw}'
                        )

        if diag_out is not None and idx <= 5:
            try:
                _cal = json.loads(ficha.get('calendario', '[]'))
            except Exception:
                _cal = []
            _precios_cal = {
                d['fecha']: d.get('precio')
                for d in _cal if d.get('disponible') and d.get('precio')
            }
            _noches = []
            _d, _fin = date.fromisoformat(fecha_entrada), date.fromisoformat(fecha_salida)
            while _d < _fin:
                _noches.append(str(_d))
                _d += timedelta(days=1)
            try:
                _suma_calendario = round(
                    sum(float(_precios_cal[n]) for n in _noches if n in _precios_cal), 2
                ) if _precios_cal else None
            except (TypeError, ValueError):
                _suma_calendario = None
            diag_out.append(
                f'[CALENDARIO DEBUG] título={ficha.get("titulo","")!r} | hotel_id={ficha.get("hotel_id")!r} | '
                f'dias_calendario={len(_cal)} | con_precio={len(_precios_cal)} | '
                f'noches_buscadas={ {n: _precios_cal.get(n) for n in _noches} } | '
                f'n_noches={len(_noches)} | precio_listado={ficha.get("precio_listado")} | '
                f'precio_total_card={ficha.get("precio_total_card")} | '
                f'suma_calendario_noches={_suma_calendario}'
            )

        return ficha

    # 4 en vez de 3: con el RF podado (~373 MB en vez de ~1,4 GB) hay más
    # margen de memoria libre para navegadores Chromium concurrentes y
    # transitorios (se lanzan y cierran por ficha, no se reutilizan).
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_procesar, row, i + 1): i
            for i, row in enumerate(listados)
        }
        for future in as_completed(futures):
            ficha = future.result()
            if ficha and not ficha.get('error'):
                fichas.append(ficha)
            procesados[0] += 1
            progreso = 0.5 + 0.5 * (procesados[0] / len(listados))
            barra.progress(progreso, text=f'{procesados[0]}/{len(listados)} alojamientos extraídos...')

    barra.progress(1.0, text='¡Listo!')
    return fichas
