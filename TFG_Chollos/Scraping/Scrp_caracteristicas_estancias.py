"""
Scrp_caracteristicas_estancias.py
==================================
Lee el CSV de URLs de Booking y extrae, para cada alojamiento:
  - Ubicación (nombre, dirección, ciudad, coordenadas, Google Maps)
  - Servicios / amenities generales
  - Servicios de la primera oferta/habitación (la más barata)
  - Calendario de disponibilidad con precio por día

La extracción combina:
  - requests + BeautifulSoup → HTML estático y GraphQL de Booking
  - Playwright (sync) → tabla de habitaciones (requiere JS)
  - ThreadPoolExecutor → paralelización por alojamiento

Uso:
    python Scraping/Scrp_caracteristicas_estancias.py
    python Scraping/Scrp_caracteristicas_estancias.py --entrada ruta/entrada.csv --salida ruta/salida.csv
"""

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

from TFG_Chollos.utils import configurar_logger

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
load_dotenv()
BASE   = Path(os.getenv("BASE") or Path(__file__).parent.parent)
logger = configurar_logger(__name__)

# Rutas de entrada y salida — solo disponibles en ejecución local
try:
    df_urls_provincias = pd.read_csv(
        BASE / "data" / "raw" / "inputs" / "urls_busqueda_booking_provincias.csv", sep="|"
    )
    urls_provincias = df_urls_provincias.set_index("localizacion")["url"].to_dict()
    INPUT_FILES  = {p: BASE / "data" / "raw" / "listados" / f"urls_booking_{p}.csv"       for p in urls_provincias}
    OUTPUT_FILES = {p: BASE / "data" / "raw" / "fichas"   / f"resultados_booking_{p}.csv" for p in urls_provincias}
except Exception:
    urls_provincias = {}
    INPUT_FILES     = {}
    OUTPUT_FILES    = {}


# =============================================================================
# CONSTANTES — CSV
# =============================================================================
CSV_SEPARATOR = "|"
COL_URL      = "url_estancia"
COL_CHECKIN  = "fecha_entrada"
COL_CHECKOUT = "fecha_salida"
COL_ADULTS   = "n_adultos"

# =============================================================================
# CONSTANTES — GRAPHQL
# =============================================================================
GRAPHQL_URL = "https://www.booking.com/dml/graphql"

# Query para obtener los servicios/amenities del alojamiento
FACILITIES_QUERY = """
query Facilities($input: HotelPageByPageNameInput!, $isPropertyFacilitiesBlockOn: Boolean = false, $facilitiesExcludeGroups: [Int!] = [], $shouldGetRelevantForYourTrip: Boolean = false, $relevantForYourTripInput: [HighlightCriterion!]! = []) {
  hotelPageByPageName(input: $input) {
    ... on HotelPageType {
      propertyDetails {
        ...PropertyFacilitiesBlockFragment @include(if: $isPropertyFacilitiesBlockOn)
        ...RelevantForYourTripFragment @include(if: $shouldGetRelevantForYourTrip)
        __typename
      }
      __typename
    }
    __typename
  }
}

fragment PropertyFacilitiesBlockFragment on Property {
  facilities(includeCommonAmenities: true, excludeGroups: $facilitiesExcludeGroups) {
    id
    groupId
    instances {
      id
      title
      attributes {
        isOffsite
        paymentInfo { chargeMode __typename }
        __typename
      }
      __typename
    }
    __typename
  }
  facilityGroups {
    id
    slug
    title
    __typename
  }
  profile {
    spokenLanguages
    __typename
  }
  __typename
}

fragment RelevantForYourTripFragment on Property {
  relevantForYourTrip: accommodationHighlights(criteria: $relevantForYourTripInput) {
    entities {
      title
      __typename
    }
    __typename
  }
  __typename
}
"""

# Query para obtener el calendario de disponibilidad y precios por día
CALENDAR_QUERY = """
query AvailabilityCalendar($input: AvailabilityCalendarQueryInput!) {
  availabilityCalendar(input: $input) {
    ... on AvailabilityCalendarQueryResult {
      hotelId
      days {
        available
        avgPriceFormatted
        checkin
        minLengthOfStay
        __typename
      }
      __typename
    }
    ... on AvailabilityCalendarQueryError {
      message
      __typename
    }
    __typename
  }
}
"""

# =============================================================================
# CONSTANTES — CABECERAS HTTP
# =============================================================================

# Cabeceras comunes a todas las peticiones
HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}

# Cabeceras para peticiones de página HTML
HEADERS_HTML = {
    **HEADERS_BASE,
    "Accept":         "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Cache-Control":  "no-cache",
}

# Cabeceras para peticiones GraphQL (JSON)
HEADERS_GQL = {
    **HEADERS_BASE,
    "Accept":       "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Origin":       "https://www.booking.com",
    "Referer":      "https://www.booking.com/",
    "X-Booking-Context-Action-Name": "hotel",
    "X-Booking-Context-Aid":         "304142",
}

# =============================================================================
# CONSTANTES — EXTRACCIÓN DE HOTEL ID
# =============================================================================

# Patrones regex ordenados de más a menos específico para extraer el hotel_id del HTML
HOTEL_ID_PATTERNS = [
    r'"hotel_id"\s*:\s*"?(\d+)"?',
    r'"hotelId"\s*:\s*"?(\d+)"?',
    r"b_hotel_id\s*[:=]\s*['\"]?(\d+)['\"]?",
    r"data-hotel-id=['\"](\d+)['\"]",
    r"hotelId%22%3A%22(\d+)",
    r'"b_hotel_id"\s*:\s*(\d+)',
    r'"b_accommodation_id"\s*:\s*(\d+)',
    r"accommodationId['\"]?\s*:\s*(\d+)",
    r"property_id['\"]?\s*:\s*(\d+)",
    r"var\s+hotelId\s*=\s*['\"]?(\d+)['\"]?",
    r'"propertyId"\s*:\s*(\d+)',
    r'"b_property_id"\s*:\s*(\d+)',
    r'property_id=(\d{5,})',
    r'"pid"\s*:\s*(\d{5,})',
    r'"id"\s*:\s*(\d{6,})',
]

# IDs internos de Booking que no corresponden a ningún alojamiento real
BOOKING_INTERNAL_IDS = {"304142", "1217750", "956449"}

MAX_WORKERS = 3   # Hilos paralelos para el ThreadPoolExecutor


# =============================================================================
# SESIÓN HTTP CON GESTIÓN AUTOMÁTICA DE WAF
# =============================================================================

class BookingSession:
    """
    Envuelve una sesión requests con resolución automática del WAF de Booking.
    - Al instanciar: abre Playwright, carga booking.com y transfiere las cookies.
    - En cada GET/POST: detecta respuestas de challenge (WAF) y refresca si es necesario.
    """

    REFRESH_EVERY = 30   # Número de peticiones tras las que se renueva la sesión

    def __init__(self):
        self.session        = requests.Session()
        self._request_count = 0
        self._refresh_lock  = threading.Lock()
        self._init_session()

    # ── Inicialización / refresco de cookies ─────────────────────────────────

    def _init_session(self):
        """Lanza Playwright, navega a booking.com y transfiere las cookies a requests."""
        print("\n  Resolviendo AWS WAF con Playwright...")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            page = browser.new_page(
                user_agent=HEADERS_BASE["User-Agent"],
                locale="es-ES",
                extra_http_headers={"Accept-Language": "es-ES,es;q=0.9"},
            )
            # Ocultamos la huella de automatización para no ser detectados
            page.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
                "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3]});"
            )
            try:
                page.goto("https://www.booking.com/", wait_until="networkidle", timeout=45_000)
            except PWTimeout:
                pass
            time.sleep(3)

            # Copiamos todas las cookies del contexto Playwright a la sesión requests
            cookies = page.context.cookies()
            self.session.cookies.clear()
            for c in cookies:
                self.session.cookies.set(
                    c["name"], c["value"],
                    domain=c.get("domain", ".booking.com"),
                    path=c.get("path", "/"),
                )
            print(f"  OK: {len(cookies)} cookies transferidas.")
            browser.close()

    def _tick(self):
        """Incrementa el contador de peticiones de forma thread-safe."""
        with self._refresh_lock:
            self._request_count += 1

    # ── Detección de WAF ──────────────────────────────────────────────────────

    @staticmethod
    def _is_waf(r: requests.Response) -> bool:
        """Devuelve True si la respuesta es un challenge del WAF de AWS."""
        return r.status_code == 202 and "challenge" in r.text.lower()

    # ── Métodos públicos de petición ──────────────────────────────────────────

    def get(self, url: str, params: dict = None) -> Optional[requests.Response]:
        """GET con reintento automático y refresco de sesión si aparece WAF."""
        self._tick()
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, headers=HEADERS_HTML, timeout=25)
                if self._is_waf(r):
                    print(f"    AVISO: WAF en GET (intento {attempt+1}) -> refrescando...")
                    self._init_session()
                    time.sleep(random.uniform(2, 4))
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException as exc:
                print(f"    [intento {attempt+1}/3] GET: {exc}")
                time.sleep(random.uniform(4, 8))
        return None

    def post_json(self, url: str, payload: dict, referer: str = "") -> Optional[dict]:
        """POST JSON (GraphQL) con reintento automático y refresco de sesión si aparece WAF."""
        self._tick()
        hdrs = {**HEADERS_GQL}
        if referer:
            hdrs["Referer"] = referer
        for attempt in range(3):
            try:
                r = self.session.post(url, json=payload, headers=hdrs, timeout=25)
                if self._is_waf(r):
                    print(f"    AVISO: WAF en POST (intento {attempt+1}) -> refrescando...")
                    self._init_session()
                    time.sleep(random.uniform(2, 4))
                    continue
                r.raise_for_status()
                data = r.json()
                if "errors" in data and not data.get("data"):
                    print(f"    AVISO: GraphQL error: {data['errors'][0].get('message','')}")
                    return None
                return data
            except (requests.RequestException, json.JSONDecodeError) as exc:
                print(f"    [intento {attempt+1}/3] POST: {exc}")
                time.sleep(random.uniform(4, 8))
        return None


# =============================================================================
# EXTRACTOR PRINCIPAL
# =============================================================================

class BookingExtractor:
    """
    Extrae toda la información de un alojamiento de Booking a partir de su URL.
    Usa BookingSession para gestionar cookies y WAF de forma automática.
    """

    def __init__(self, currency: str = "EUR", language: str = "es"):
        self.currency = currency
        self.language = language
        self.http     = BookingSession()   # Abre sesión y resuelve WAF al instanciar

    @staticmethod
    def _delay(a: float = 1.5, b: float = 3.5):
        """Pausa aleatoria para no saturar el servidor."""
        time.sleep(random.uniform(a, b))

    # ── Helpers de URL ────────────────────────────────────────────────────────

    @staticmethod
    def _pagename(url: str) -> str:
        """Extrae el slug del alojamiento de la URL de Booking."""
        m = re.search(r"/hotel/[a-z]{2}/([^.?/]+)", url)
        return m.group(1) if m else ""

    @staticmethod
    def _country(url: str) -> str:
        """Extrae el código de país de 2 letras de la URL de Booking."""
        m = re.search(r"/hotel/([a-z]{2})/", url)
        return m.group(1) if m else "es"

    # ── Extracción del hotel_id ───────────────────────────────────────────────

    def _get_hotel_id(self, html_text: str, pagename: str, country_code: str) -> Optional[str]:
        """
        Busca el hotel_id de Booking con tres estrategias en cascada:
          1. Regex sobre el HTML crudo
          2. JSON embebido en __NEXT_DATA__ (Next.js)
          3. Fallback: nueva petición HTTP con URL canónica
        """
        # 1. Patrones regex sobre el HTML crudo
        for pattern in HOTEL_ID_PATTERNS:
            m = re.search(pattern, html_text)
            if m and len(m.group(1)) >= 4 and m.group(1) not in BOOKING_INTERNAL_IDS:
                return m.group(1)

        # 2. JSON embebido en __NEXT_DATA__ (apps React/Next.js)
        hid = self._get_hotel_id_from_next_data(html_text)
        if hid:
            return hid

        # 3. Fallback: nueva petición con URL canónica
        if pagename:
            hid = self._hotel_id_graphql(pagename, country_code)
            if hid:
                return hid

        return None

    def _get_hotel_id_from_next_data(self, html_text: str) -> Optional[str]:
        """Busca el hotel_id dentro del JSON inyectado por Next.js en el HTML."""
        try:
            soup   = BeautifulSoup(html_text, 'lxml')
            script = soup.find('script', id='__NEXT_DATA__')
            if script and script.string:
                return self._search_dict_for_hotel_id(json.loads(script.string))
        except Exception:
            pass
        return None

    def _search_dict_for_hotel_id(self, obj, depth: int = 0) -> Optional[str]:
        """Recorre recursivamente un dict/list buscando un campo que contenga el hotel_id."""
        if depth > 12:
            return None
        if isinstance(obj, dict):
            for key in ('hotel_id', 'hotelId', 'propertyId', 'b_accommodation_id',
                        'accommodation_id', 'b_hotel_id', 'b_property_id', 'pid'):
                val = obj.get(key)
                if val:
                    s = str(val)
                    if s.isdigit() and len(s) >= 4 and s not in BOOKING_INTERNAL_IDS:
                        return s
            for v in obj.values():
                result = self._search_dict_for_hotel_id(v, depth + 1)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj[:30]:
                result = self._search_dict_for_hotel_id(item, depth + 1)
                if result:
                    return result
        return None

    def _hotel_id_graphql(self, pagename: str, country_code: str) -> Optional[str]:
        """Intenta obtener el hotel_id haciendo una petición HTTP a la URL canónica."""
        fallback_url = (
            f"https://www.booking.com/hotel/{country_code}/{pagename}.es.html"
            f"?aid=304142&lang=es"
        )
        r = self.http.get(fallback_url)
        if r is None:
            return None
        for pattern in HOTEL_ID_PATTERNS:
            m = re.search(pattern, r.text)
            if m and len(m.group(1)) >= 4 and m.group(1) not in BOOKING_INTERNAL_IDS:
                return m.group(1)
        return None

    # ── 1. UBICACIÓN ──────────────────────────────────────────────────────────

    @staticmethod
    def _location(soup: BeautifulSoup, html_text: str) -> dict:
        """
        Extrae nombre, dirección, coordenadas y enlace a Google Maps del HTML.
        Primero intenta JSON-LD (schema.org), luego selectores CSS de fallback.
        """
        loc = {
            "nombre_booking": "", "direccion": "", "ciudad": "",
            "pais": "", "codigo_postal": "", "latitud": "", "longitud": "",
            "google_maps": "",
        }

        # Nombre del alojamiento: probamos varios selectores hasta encontrar uno
        for sel in ["h2.pp-header__name", "h1.pp-header__name",
                    '[data-testid="property-header-name"]', "h1"]:
            t = soup.select_one(sel)
            if t and t.get_text(strip=True):
                loc["nombre_booking"] = t.get_text(strip=True)
                break

        # Dirección y coordenadas: leemos el JSON-LD embebido (schema.org)
        for s in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                obj  = json.loads(s.string or "")
                geo  = obj.get("geo") or {}
                addr = obj.get("address") or {}
                if geo or addr:
                    loc.update({
                        "latitud":       str(geo.get("latitude",  "")),
                        "longitud":      str(geo.get("longitude", "")),
                        "direccion":     addr.get("streetAddress", ""),
                        "ciudad":        addr.get("addressLocality", ""),
                        "pais":          addr.get("addressCountry", ""),
                        "codigo_postal": addr.get("postalCode", ""),
                    })
                    if not loc["nombre_booking"]:
                        loc["nombre_booking"] = obj.get("name", "")
            except Exception:
                pass

        # Fallback de dirección si el JSON-LD no la tenía
        if not loc["direccion"]:
            for sel in ['[data-testid="address"]', ".hp_address_subtitle",
                        'span[itemprop="streetAddress"]']:
                t = soup.select_one(sel)
                if t:
                    loc["direccion"] = t.get_text(strip=True)
                    break

        # Fallback de coordenadas vía regex sobre el HTML crudo
        if not loc["latitud"]:
            for p in [r'"latitude"\s*:\s*([-\d.]+)', r"b_map_center_lat\s*=\s*([-\d.]+)"]:
                m = re.search(p, html_text)
                if m:
                    loc["latitud"] = m.group(1)
                    break
        if not loc["longitud"]:
            for p in [r'"longitude"\s*:\s*([-\d.]+)', r"b_map_center_lon\s*=\s*([-\d.]+)"]:
                m = re.search(p, html_text)
                if m:
                    loc["longitud"] = m.group(1)
                    break

        # Construimos el enlace a Google Maps si tenemos coordenadas
        if loc["latitud"] and loc["longitud"]:
            loc["google_maps"] = (
                f"https://www.google.com/maps?q={loc['latitud']},{loc['longitud']}"
            )
        return loc

    # ── 2. SERVICIOS GENERALES ────────────────────────────────────────────────

    def _amenities(self, pagename: str, country_code: str,
                   checkin: str, checkout: str, adults: int,
                   soup: BeautifulSoup) -> list:
        """
        Obtiene los servicios/amenities del alojamiento vía GraphQL.
        Si la API no devuelve datos, hace fallback a scraping del HTML con BeautifulSoup.
        """
        found = set()

        # Construimos el payload para la query de servicios
        payload = {
            "operationName": "Facilities",
            "query": FACILITIES_QUERY,
            "variables": {
                "isPropertyFacilitiesBlockOn": True,
                "shouldGetRelevantForYourTrip": True,
                "facilitiesExcludeGroups": [37, 38, 39, 40, 41],
                "relevantForYourTripInput": [
                    {"criterion": "relevantForYourTrip", "criterionParams": {"limit": 10}}
                ],
                "input": {
                    "pageNameDetails": {"countryCode": country_code, "pagename": pagename},
                    "searchConfig": {
                        "searchConfigDate": {"checkin": checkin, "checkout": checkout},
                        "nbRooms": 1, "nbAdults": adults, "nbChildren": 0, "childrenAges": [],
                    },
                    "selectedFilters": "",
                },
            },
        }

        referer = f"https://www.booking.com/hotel/{country_code}/{pagename}.es.html"
        data    = self.http.post_json(GRAPHQL_URL, payload, referer=referer)

        if data:
            hpbpn = data.get("data", {}).get("hotelPageByPageName", {})
            if isinstance(hpbpn, list):
                hpbpn = hpbpn[0] if hpbpn else {}

            prop = hpbpn.get("propertyDetails", {}) or {}
            if isinstance(prop, list):
                prop = prop[0] if prop else {}

            # Recorremos los grupos de servicios y sus instancias
            for fac in prop.get("facilities", []) or []:
                if not isinstance(fac, dict):
                    continue
                for instance in fac.get("instances", []) or []:
                    if not isinstance(instance, dict):
                        continue
                    title = instance.get("title", "").strip()
                    if title and len(title) > 2:
                        found.add(title)

            # También recogemos los idiomas hablados del perfil del alojamiento
            profile = prop.get("profile", {}) or {}
            if isinstance(profile, list):
                profile = profile[0] if profile else {}
            for lang in profile.get("spokenLanguages", []) or []:
                if lang and len(lang) > 2:
                    found.add(lang)

            # Y los highlights de "relevant for your trip"
            rft_raw = prop.get("relevantForYourTrip", []) or []
            if isinstance(rft_raw, dict):
                rft_raw = [rft_raw]
            for rft_item in rft_raw:
                if not isinstance(rft_item, dict):
                    continue
                for entity in rft_item.get("entities", []) or []:
                    if not isinstance(entity, dict):
                        continue
                    title = entity.get("title", "").strip()
                    if title and len(title) > 2:
                        found.add(title)

        # Fallback: si GraphQL no devolvió nada, scrapeamos el HTML directamente
        if not found:
            for wrapper in soup.select('[data-testid="property-most-popular-facilities-wrapper"]'):
                for li in wrapper.select("li"):
                    icon = li.select_one('[data-testid="facility-icon"]')
                    if icon and icon.parent:
                        for child in icon.parent.children:
                            if child == icon:
                                continue
                            if hasattr(child, "get_text"):
                                t = child.get_text(strip=True)
                                if t and len(t) > 2:
                                    found.add(t)
                                    break

        return sorted(found)

    # ── 3. SERVICIOS DE LA PRIMERA OFERTA/HABITACIÓN ─────────────────────────

    def _room_amenities(self, url: str) -> list:
        """
        Extrae los servicios de la primera oferta (la más barata) usando Playwright,
        que ejecuta JavaScript y espera a que cargue la tabla de habitaciones.
        Reutiliza las cookies de la sesión HTTP ya establecida para no resolver el WAF otra vez.
        """
        found = []

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                user_agent=HEADERS_BASE["User-Agent"],
                locale="es-ES",
            )
            context.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
                "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3]});"
            )

            # Transferimos las cookies de requests a Playwright para no resolver el WAF otra vez
            cookies_playwright = [
                {"name": c.name, "value": c.value, "domain": ".booking.com", "path": "/"}
                for c in self.http.session.cookies
            ]
            context.add_cookies(cookies_playwright)

            page = context.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=45_000)
                # Esperamos a que aparezca la tabla de habitaciones (requiere JS)
                page.wait_for_selector("tr.hprt-table-cheapest-block", timeout=15_000)
            except PWTimeout:
                print("    AVISO: Timeout esperando tabla de habitaciones")
                browser.close()
                return []

            html = page.content()
            browser.close()

        soup = BeautifulSoup(html, "lxml")

        # La clase hprt-table-cheapest-block identifica siempre la oferta más barata
        primera_oferta = soup.select_one("tr.hprt-table-cheapest-block")
        if not primera_oferta:
            print("    AVISO: No se encontró la primera oferta en el HTML renderizado")
            return []

        # Extraemos los badges de servicios de esa primera oferta
        for fac in primera_oferta.select(".hprt-facilities-facility"):
            nombre = fac.get("data-name-en", "").strip()
            # "room size" es el tipo, no el valor — usamos el texto visible ("44 m²")
            if nombre.lower() == "room size" or not nombre:
                nombre = fac.get_text(strip=True)
            if nombre and len(nombre) > 2:
                found.append(nombre)

        return sorted(set(found))

    # ── 4. CALENDARIO DE DISPONIBILIDAD ──────────────────────────────────────

    def _calendar(self, hotel_id: str, pagename: str, country_code: str,
                  checkin: str, checkout: str, adults: int) -> list:
        """
        Consulta el calendario de disponibilidad y precios por día vía GraphQL.
        Pide 365 días en chunks de 90 para no saturar la API.
        """
        from datetime import timedelta

        today      = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        referer    = f"https://www.booking.com/hotel/{country_code}/{pagename}.es.html"
        chunk_days = 90
        total_days = 365
        all_days   = {}

        cursor            = today
        fetched           = 0
        consecutive_empty = 0

        while fetched < total_days:
            start_date  = cursor.strftime("%Y-%m-%d")
            amount_days = min(chunk_days, total_days - fetched)

            payload = {
                "operationName": "AvailabilityCalendar",
                "query": CALENDAR_QUERY,
                "variables": {
                    "input": {
                        "travelPurpose": 2,
                        "pagenameDetails": {"countryCode": country_code, "pagename": pagename},
                        "searchConfig": {
                            "searchConfigDate": {
                                "startDate":    start_date,
                                "amountOfDays": amount_days,
                            },
                            "nbAdults": adults, "nbChildren": 0,
                            "nbRooms": 1, "childrenAges": [],
                        },
                    }
                },
            }

            data     = self.http.post_json(GRAPHQL_URL, payload, referer=referer)
            days_raw = []
            if data:
                days_raw = (
                    data.get("data", {})
                        .get("availabilityCalendar", {})
                        .get("days", [])
                )

            # Si dos chunks consecutivos vienen vacíos, el hotel no tiene más disponibilidad
            if not days_raw:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
            else:
                consecutive_empty = 0
                for d in days_raw:
                    precio_raw = d.get("avgPriceFormatted", "")
                    precio_num = re.sub(r"[^\d.]", "", precio_raw)
                    fecha = d.get("checkin", "")
                    if fecha:
                        all_days[fecha] = {
                            "fecha":           fecha,
                            "disponible":      d.get("available", False),
                            "precio":          precio_num if precio_num and precio_num != "0" else "",
                            "precio_raw":      precio_raw,
                            "estancia_minima": d.get("minLengthOfStay", 1),
                        }

            fetched += amount_days
            cursor  += timedelta(days=amount_days)
            time.sleep(random.uniform(0.3, 0.8))

        return sorted(all_days.values(), key=lambda x: x["fecha"])

    # ── PROCESAR UNA FILA ─────────────────────────────────────────────────────

    def process_row(self, row: dict, idx: int = 0, total: int = 0) -> dict:
        """
        Procesa un alojamiento completo: ubicación, servicios, habitación y calendario.
        Devuelve un dict con todos los campos extraídos listo para escribir al CSV.
        """
        url      = row[COL_URL].strip()
        checkin  = row[COL_CHECKIN].strip()
        checkout = row[COL_CHECKOUT].strip()
        try:
            adults = int(row.get(COL_ADULTS, 2) or 2)
        except ValueError:
            adults = 2

        sep = "-" * 46
        tag = f"[{idx}/{total}] " if idx else ""
        print(f"\n{tag}{sep}")
        print(f"  Alojamiento : {row.get('titulo', '')}")
        print(f"  Periodo     : {checkin} -> {checkout}  |  Adultos: {adults}")

        pagename     = self._pagename(url)
        country_code = self._country(url)

        # Descargamos el HTML principal del alojamiento
        r = self.http.get(url)
        if r is None:
            return {**row, "hotel_id": "", "nombre_booking": "",
                    "direccion": "", "ciudad": "", "pais": "",
                    "codigo_postal": "", "latitud": "", "longitud": "",
                    "google_maps": "", "servicios": "[]",
                    "servicios_habitacion": "[]",
                    "calendario": "[]", "error": "No se pudo cargar la página"}

        soup      = BeautifulSoup(r.text, "lxml")
        html_text = r.text

        hotel_id = self._get_hotel_id(html_text, pagename, country_code)
        print(f"  hotel_id    : {hotel_id or 'no encontrado'}  |  slug: {pagename}")

        print("  -> Ubicación...")
        loc = self._location(soup, html_text)

        print("  -> Servicios generales...")
        amenities = self._amenities(pagename, country_code, checkin, checkout, adults, soup)

        print("  -> Servicios de la primera oferta...")
        room_amenities = self._room_amenities(url)
        print(f"     {len(room_amenities)} servicios encontrados en la primera oferta.")

        calendar = []
        if hotel_id:
            print("  -> Calendario...")
            calendar = self._calendar(hotel_id, pagename, country_code, checkin, checkout, adults)
            disponibles = sum(1 for d in calendar if d["disponible"])
            print(f"     {len(calendar)} días — {disponibles} con disponibilidad y precio.")
        else:
            print("  AVISO: Sin hotel_id -> calendario omitido.")

        return {
            **row,
            "hotel_id":               hotel_id or "",
            "nombre_booking":         loc["nombre_booking"],
            "direccion":              loc["direccion"],
            "ciudad":                 loc["ciudad"],
            "pais":                   loc["pais"],
            "codigo_postal":          loc["codigo_postal"],
            "latitud":                loc["latitud"],
            "longitud":               loc["longitud"],
            "google_maps":            loc["google_maps"],
            "servicios":              json.dumps(amenities,      ensure_ascii=False),
            "servicios_habitacion":   json.dumps(room_amenities, ensure_ascii=False),
            "calendario":             json.dumps(calendar,       ensure_ascii=False),
            "fecha_extraccion_ficha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error":                  "",
        }

    # ── PROCESAR CSV ──────────────────────────────────────────────────────────

    @staticmethod
    def _urls_ya_procesadas(output_path: str) -> set:
        """
        Lee el CSV de salida si ya existe y devuelve las URLs ya procesadas.
        Permite reanudar el scraping sin reprocesar alojamientos ya extraídos.
        """
        ya_hechas = set()
        try:
            with open(output_path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter=CSV_SEPARATOR)
                for row in reader:
                    url = row.get(COL_URL, "").strip()
                    if url:
                        ya_hechas.add(url)
            print(f"  Reanudando: {len(ya_hechas)} alojamientos ya procesados, se omiten.")
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"  AVISO: No se pudo leer el CSV de salida para reanudar: {exc}")
        return ya_hechas

    @staticmethod
    def _read_csv(filepath: str) -> list:
        """Lee el CSV de URLs de entrada y devuelve una lista de dicts."""
        try:
            with open(filepath, encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f, delimiter=CSV_SEPARATOR))
            print(f"  CSV leído: {len(rows)} filas  |  sep='{CSV_SEPARATOR}'")
            return rows
        except Exception as exc:
            print(f"ERROR leyendo CSV: {exc}")
            return []

    def process_csv(self, input_path: str, output_path: str, workers: int = MAX_WORKERS):
        """
        Procesa todos los alojamientos del CSV de entrada en paralelo.
        Escribe los resultados en el CSV de salida en orden de llegada.
        Es reanudable: si el CSV de salida ya existe, omite las URLs ya procesadas.
        """
        rows = self._read_csv(input_path)
        if not rows:
            print("CSV vacío o ilegible.")
            return

        # Verificamos que el CSV tiene las columnas necesarias
        missing = {COL_URL, COL_CHECKIN, COL_CHECKOUT, COL_ADULTS} - set(rows[0].keys())
        if missing:
            print(f"Faltan columnas: {missing}")
            return

        # Filtramos las filas ya procesadas en ejecuciones anteriores
        ya_hechas       = self._urls_ya_procesadas(output_path)
        rows_pendientes = [r for r in rows if r[COL_URL].strip() not in ya_hechas] if ya_hechas else rows

        total_original = len(rows)
        total          = len(rows_pendientes)

        print(f"\n{'='*62}")
        print(f"  BOOKING EXTRACTOR --- {total_original} alojamientos en total")
        print(f"  Pendientes  : {total}  |  Ya hechos: {total_original - total}")
        print(f"  Workers     : {workers}")
        print(f"  Entrada : {input_path}")
        print(f"  Salida  : {output_path}")
        print(f"{'='*62}\n")

        if not rows_pendientes:
            print("  Todo ya estaba procesado. Nada que hacer.")
            return

        lock          = threading.Lock()
        errores       = [0]
        results_map   = {}
        next_to_write = [0]
        writer        = [None]
        csv_file      = [None]

        try:
            # "a" = append si hay filas ya hechas, "w" = nuevo fichero si es la primera vez
            modo         = "a" if ya_hechas else "w"
            escribir_hdr = not ya_hechas
            csv_file[0]  = open(output_path, modo, newline="", encoding="utf-8-sig")

            def process_and_store(idx, row):
                # Escalonamos el inicio de cada hilo para no lanzarlos todos a la vez
                time.sleep(idx % workers * random.uniform(1.2, 2.5))
                try:
                    result = self.process_row(row, idx + 1, total)
                except Exception as exc:
                    print(f"  Error [{idx+1}]: {exc}")
                    result = {**row, "error": str(exc)}
                    with lock:
                        errores[0] += 1

                # Guardamos el resultado y escribimos en orden secuencial
                with lock:
                    results_map[idx] = result
                    while next_to_write[0] in results_map:
                        r = results_map.pop(next_to_write[0])
                        if writer[0] is None:
                            writer[0] = csv.DictWriter(
                                csv_file[0], fieldnames=list(r.keys()),
                                extrasaction="ignore", delimiter="|",
                            )
                            if escribir_hdr:
                                writer[0].writeheader()
                        writer[0].writerow(r)
                        csv_file[0].flush()
                        next_to_write[0] += 1

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_and_store, i, row): i
                           for i, row in enumerate(rows_pendientes)}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"  Future error: {exc}")

        finally:
            if csv_file[0]:
                csv_file[0].close()

        print(f"\n{'='*62}")
        print(f"  Completado --- {total} procesados  |  {errores[0]} errores")
        print(f"  CSV: {output_path}")
        print(f"{'='*62}")


# =============================================================================
# CLI
# =============================================================================

def main():
    comienzo = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", default=None)
    parser.add_argument("--salida",  default=None)
    parser.add_argument("--moneda",  default="EUR")
    args = parser.parse_args()

    extractor = BookingExtractor(currency=args.moneda)

    if args.entrada and args.salida:
        # Modo manual: procesa solo el par entrada/salida indicado por argumento
        extractor.process_csv(args.entrada, args.salida)
    else:
        # Modo automático: itera sobre todas las provincias definidas en INPUT_FILES
        for lugar in INPUT_FILES:
            print(f"\n{'='*62}")
            print(f"  LUGAR: {lugar}")
            print(f"{'='*62}")
            extractor.process_csv(str(INPUT_FILES[lugar]), str(OUTPUT_FILES[lugar]))

    duracion = datetime.now() - comienzo
    logger.info(f"Scraping completado. Inicio: {comienzo.strftime('%Y-%m-%d %H:%M:%S')} | Duracion: {duracion}")


if __name__ == "__main__":
    main()
