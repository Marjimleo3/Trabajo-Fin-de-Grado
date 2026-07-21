"""
Extrae info básica de estancias en Booking.com para cada una de las provincias de Andalucía.

Automatiza la extracción de datos de alojamientos en Booking.com mediante Selenium, simulando navegación real (scroll dinámico hasta agotar resultados, cierre de popups) y optimizando el rendimiento al bloquear recursos innecesarios (imágenes, fuentes, analytics).

Datos extraídos
---------------
Por cada estancia:
    - lugar              : Ciudad o zona de búsqueda
    - titulo             : Nombre del alojamiento
    - url_estancia       : URL de la ficha del alojamiento
    - valoracion_clientes: Puntuación media de los clientes
    - n_valoraciones     : Número total de valoraciones
    - tipo               : Tipo de alojamiento (hotel, apartamento, etc.)
    - estrellas          : Categoría en estrellas

Dependencias
------------
    selenium webdriver-manager beautifulsoup4 pandas python-dotenv --active --link-mode=copy

Entrada / Salida
----------------
    Entrada : Generador_urls_generales.py
                - urls_provincias : Lista de URLs de búsqueda por provincia
                - FECHA_ENTRADA   : Fecha de check-in
                - FECHA_SALIDA    : Fecha de check-out
                - N_ADULTOS       : Número de adultos
                - N_HABITACIONES  : Número de habitaciones
                - N_MENORES       : Número de menores

    Salida  : BASE / "data" / "Booking" / "urls_booking_{lugar}.csv"

*Nuevo: Hemos añadido la generalización de rutas 
*Nuevo: Solucionado el problema del cap de 800 estancias a través de un sistema de partición de resultados añadiendo filtros a las búsquedas   
"""
# =============================================================================
# IMPORTS
# =============================================================================
#Librerías estándar (vienen incluidas con Python):
import time
import random
import re
from datetime import datetime

#Librerías de terceros (es necesario instalarlas):
import logging
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

#Módulos propios del proyecto
from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG
from TFG_Chollos.Scraping.Generador_urls_generales import (
    FECHA_ENTRADA, FECHA_SALIDA,
    N_ADULTOS, N_HABITACIONES, N_MENORES
)

# =============================================================================
# CONSTANTES
# =============================================================================


# =============================================================================
# CONFIGURACIÓN DEL LOGGER
# =============================================================================
logger = configurar_logger(__name__)

# =============================================================================
# FUNCIONES
# =============================================================================
def crear_driver() -> webdriver.Chrome:
    '''
    Crea y devuelve un driver de Chrome configurado con opciones anti-scraping y de optimización.
    '''
    logging.getLogger("WDM").setLevel(logging.WARNING)  # Silencia los logs de webdriver-manager
    options = Options()

    #Opciones anti-scraping:
    options.add_argument("--disable-blink-features=AutomationControlled")   #Oculta una señal que da Chrome cuando se usa con Selenium
    options.add_experimental_option("excludeSwitches", ["enable-automation"])   #Al cargar Chrome con Selenium, no añadir --enable-automation, porque provoca que arriba del navegador aparezca 'Chrome is being controlled by automated test software'
    options.add_experimental_option("useAutomationExtension", False)   #Que Chrome no cargue una extensión visible para anti-scraping

    #Opciones de optimización:
    #CON ESTO NO FUNCIONA options.add_argument("--headless=new")   # modo headless (no habre la ventana del navegador)
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        # "profile.managed_default_content_settings.stylesheets": 2,   No bloquear CSS, ya que pierdes más estabilidad de lo que ganas en velocidad
        "profile.managed_default_content_settings.fonts": 2
    }
    options.add_experimental_option("prefs", prefs)   # evita cargar imágenes y los tipos de letra
    options.add_argument("--disable-gpu")   # desactiva GPU
    options.add_argument("--disable-extensions")   # desactiva extensiones
    options.add_argument("--disable-notifications")   # desactiva notificaciones
    options.add_argument("--no-sandbox")   # evitar sandbox (reduce consumo en algunos casos)
    options.add_argument("--disable-dev-shm-usage")   # desactivar dev-shm
    options.page_load_strategy = "eager"   # Espera solo a que esté listo el html y el DOM

    #Para mostrar bien el funcionamiento del código, debemos descomentar el siguiente parámetro, y comentar las 2 primeras opciones de optimización ("--headless=new" y "prefs")
    # options.add_argument("--start-maximized")

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )


def cerrar_popup(driver):
        '''
        Cierra el popup de publicidad que aparece al iniciar la página de Booking.com
        '''

        try:
            popup = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[aria-modal="true"]'))
            )
            boton = popup.find_element(By.CSS_SELECTOR, 'button[aria-label]')
            boton.click()
            print('[OK] Popup cerrado correctamente')
        except Exception:
            print('[ERROR] No hay Popup')
            pass


def cargar_todo(driver):
    '''
    Ejecuta el siguiente bucle:
    Scroll hasta el final de la página -> Pulsa el botón 'Cargar más'.
    El bucle termina cuando no encuentra el botón 'Cargar más' o cuando no se generan nuevos alojamientos.
    '''
    # Scroll inicial doble
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)

    intentos_sin_boton = 0  # tolerancia antes de rendirse

    while True:
        try:
            # boton = driver.find_element(By.XPATH, "//button[.//span[contains(text(),'Cargar más')]]")
            boton = WebDriverWait(driver, 10).until(   # Espera activa hasta 6s a que el botón esté clicable
                EC.element_to_be_clickable((By.XPATH, "//button[.//span[contains(text(),'Cargar más')]]"))
            )

            tarjetas_antes = len(driver.find_elements(By.CSS_SELECTOR, '[data-testid="property-card"]'))   # Contar tarjetas antes del clic

            driver.execute_script("arguments[0].scrollIntoView();", boton)
            time.sleep(0.5)
            boton.click()
            print("       Botón pulsado")

            WebDriverWait(driver, 8).until(   # Esperar a que aparezcan MÁS tarjetas (hasta 8s)
                lambda d: len(d.find_elements(By.CSS_SELECTOR, '[data-testid="property-card"]')) > tarjetas_antes
            )

            intentos_sin_boton = 0  # reset al tener éxito

        except TimeoutException:
            intentos_sin_boton += 1
            print(f"       Timeout esperando botón/carga ({intentos_sin_boton}/3)")
            time.sleep(3)   # pausa antes de reintentar
            if intentos_sin_boton >= 3:
                print("       No hay más botones, Fin.")
                break


def filtro_precio(url, precio_min, precio_max):
    '''
    Añade un filtro de rango de precio a una url de búsqueda de booking
    '''
    filtro = f"&nflt=price%3DEUR-{precio_min}-{precio_max}-1"
    return url + filtro


def n_alojamientos(driver):
    '''
    Obtiene, con Selenium, el número de alojamientos de la búsqueda en Booking.com
    '''
    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        titulo = soup.find('h1')
        span  = titulo.find('span') if titulo else None
        texto = (span if span else titulo).get_text(strip=True)
        return int(re.search(r"[\d.]+", texto).group().replace(".", ""))
    except Exception as e:
        print(f"[WARN] No se pudo leer el total de resultados: {e}")
    return 0


def extraer_alojamientos(driver, clave):
    '''
    Identifica y extrae las características más relevante para el estudio, de cada una de las tarjetas de alojamientos en los resultados de búsqueda
    '''
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    estancias = soup.find_all('div', {"data-testid":"property-card"})
    print("Número total de estancias:", len(estancias))

    alojamientos = []
    for estancia in estancias:

        try:
            enlace = estancia.find('a', {"data-testid":"title-link"})
            enlace_limpio = enlace['href'].split('?')[0]   #Eliminamos datos innecesarios de la url para evitar bloqueos por automatización
            url = f'{enlace_limpio}?checkin={FECHA_ENTRADA}&checkout={FECHA_SALIDA}&group_adults={N_ADULTOS}&req_adults={N_ADULTOS}&no_rooms={N_HABITACIONES}&group_children={N_MENORES}&req_children={N_MENORES}'

            nombre = estancia.find('div', {"data-testid":"title"})
            titulo = nombre.get_text(strip=True)   #strip=True elimina espacios en blanco al principio y al final

            reviews = estancia.find('div', {"data-testid":"review-score"})
            if reviews:   #Usamos regex (Expresiones regulares) y no usamos 'class' para extraer información, porque cambian constantemente
                valoracion_clientes = re.search(r"\d+(?:[.,]\d)?", reviews.get_text()).group()   #Estructura regex: re.search(patrón, texto). group() extrae el texto del match
                numero_valoraciones = re.search(r"\d[\d.]*\s+comentarios?", reviews.get_text()).group()
                n_valoraciones = numero_valoraciones.replace(".", "").split(" ")[0]
            else:
                valoracion_clientes = 'NA'
                n_valoraciones = 'NA'

            if estancia.find('div', {"data-testid":"rating-squares"}):
                tipo = 'Otro'
                n_estrellas = estancia.find('div', {"data-testid":"rating-squares"})   #Cuadrados = estrellas de booking
                estrellas = len(n_estrellas.find_all('div', recursive=False))   #Recursive=F especifica los hijos Directos
            elif estancia.find('div', {"data-testid":"rating-stars"}):
                tipo = 'Hotel'
                n_estrellas = estancia.find('div', {"data-testid":"rating-stars"})
                estrellas = len(n_estrellas.find_all('div', recursive=False))   #Recursive=F especifica los hijos Directos
            else:
                tipo = 'NA'
                n_estrellas = 'NA'
                estrellas = 'NA'

            alojamientos.append({
                'lugar': clave,
                'titulo': titulo,
                'url_estancia': url,
                'valoracion_clientes': valoracion_clientes,
                'n_valoraciones': n_valoraciones,
                'tipo': tipo,
                'estrellas': estrellas,
                'fecha_extraccion_listado': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        except Exception as e:
            print(f"Error extrayendo la URL de {estancia}. Error {e}")
            pass

    return alojamientos


def scrape_con_subdivision(driver, clave, url, precio_min=0, precio_max=6000, profundidad=0):   #Precio por noche
    '''
    Particiona a través de filtros de precio, los resultados de búsqueda si estos superan los 750 resultados.
    '''
    url_filtrada = filtro_precio(url, precio_min, precio_max)
    driver.get(url_filtrada)
    cerrar_popup(driver)

    total = n_alojamientos(driver)

    if total == 0:
        return []

    if total <= 750:
        cargar_todo(driver)
        return extraer_alojamientos(driver, clave)

    # Supera 750 → partir el rango por la mitad y recursar
    mid = (precio_min + precio_max) // 2
    mitad_baja = scrape_con_subdivision(driver, clave, url, precio_min, mid, profundidad + 1)
    mitad_alta = scrape_con_subdivision(driver, clave, url, mid, precio_max, profundidad + 1)
    return mitad_baja + mitad_alta

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():

    comienzo = datetime.now()

    # Cargamos las URLs de búsqueda generadas por Generador_urls_generales.py
    BASE = conseguir_ruta_general_TFG()
    df_urls_provincias = pd.read_csv(BASE / "data" / "raw" / "inputs" / "urls_busqueda_booking_provincias.csv", sep="|")
    urls_provincias = df_urls_provincias.set_index("localizacion")["url"].to_dict()

    driver = crear_driver()

    try:
        for clave, url in urls_provincias.items():
            print(f'\n{"="*50}')
            print(f'Scrapeando {clave}')
            print(f'{"="*50}')

            # Scraping con subdivisión por precio para superar el límite de 750 resultados
            alojamientos_raw = scrape_con_subdivision(driver, clave, url)

            # Deduplicamos por URL base (sin parámetros) para eliminar alojamientos repetidos
            vistos = set()
            alojamientos = []
            for a in alojamientos_raw:
                url_limpia = a['url_estancia'].split('?')[0]
                if url_limpia not in vistos:
                    vistos.add(url_limpia)
                    alojamientos.append(a)

            logger.info(f'[OK] Total único tras deduplicar: {len(alojamientos)} alojamientos')

            # Añadimos las fechas y parámetros de búsqueda al DataFrame y guardamos el CSV
            df = pd.DataFrame(alojamientos)
            df['fecha_entrada'], df['fecha_salida'], df['n_adultos'], df['n_habitaciones'], df['n_menores'] = FECHA_ENTRADA, FECHA_SALIDA, N_ADULTOS, N_HABITACIONES, N_MENORES
            df = df.to_csv(BASE / "data" / "raw" / "listados" / f"urls_booking_{clave}.csv", index=False, sep='|')
            logger.info('[OK] Datos guardados correctamente')

            # Pausa aleatoria entre provincias para no saturar el servidor
            time.sleep(random.uniform(8, 15))

    finally:
        driver.quit()
        duracion = datetime.now() - comienzo
        logger.info(f"Scraping completado. Inicio: {comienzo.strftime('%Y-%m-%d %H:%M:%S')} | Duración: {duracion}")
    

if __name__ == "__main__":
    main()