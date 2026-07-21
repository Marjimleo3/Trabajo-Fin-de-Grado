"""
Generador_urls_generales.py
==============================================================================
Define y guarda las variables necesarias para la búsqueda en booking.

Dependencias:
    - Python >= 3.10
    - pandas >= 3.0.1

Requisitos:
    uv
    uv add pandas --active --link-mode=copy

Uso:
    python booking_extractor.py --output data_Booking/info_estatica/
"""

# =============================================================================
# IMPORTS
# =============================================================================
#Librerías de terceros (es necesario instalarlas):
import pandas as pd
from dotenv import load_dotenv

#Módulos propios del proyecto
from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG

# =============================================================================
# CONSTANTES
# =============================================================================
FECHA_ENTRADA = '2026-09-16'
FECHA_SALIDA = '2026-09-19'
N_ADULTOS = 2
N_HABITACIONES = 1
N_MENORES = 0

PROVINCIAS = {
    'Huelva_Provincia': ['Huelva', '758'],
    'Sevilla_Provincia': ['Sevilla', '774'],
    'Cádiz_Provincia': ['Cádiz', '747'],
    'Jaén_Provincia': ['Jaén', '759'],
    'Granada_Provincia': ['Granada', '755'],
    'Almería_Provincia': ['Almería', '1363'],
    'Córdoba_Provincia': ['Córdoba', '750'],
    'Málaga_Provincia': ['Málaga', '766']
    }

LUGARES = {
    'Punta_Umbría': ['Punta+Umbría', '12834'],
    'Rota': ['Rota','-399726'],
    'Playa_Malagueta':['Playa+de+la+Malagueta','54870'],
    'Playa_Marbella':['marbella',-391076]
    }

# =============================================================================
# CONFIGURACIÓN DEL LOGGER
# =============================================================================
logger = configurar_logger(__name__)

# =============================================================================
# FUNCIONES
# =============================================================================
def generador_urls(FECHA_ENTRADA:str, FECHA_SALIDA:str, N_ADULTOS:int, N_HABITACIONES:int, N_MENORES:int, LUGARES:dict, PROVINCIAS:dict) -> tuple:
    """
    Genera las URLs de búsqueda de Booking para los lugares y provincias definidos previamente.

    Args:
        FECHA_ENTRADA (str): Fecha de entrada en formato YYYY-MM-DD.
        FECHA_SALIDA (str): Fecha de salida en formato YYYY-MM-DD.
        N_ADULTOS (int): Número de adultos.
        N_HABITACIONES (int): Número de habitaciones.
        N_MENORES (int): Número de menores.
        lugares (dict): Diccionario con los lugares y sus IDs de Booking.
        provincias (dict): Diccionario con las provincias y sus IDs de Booking.

    Returns:
        tuple: (urls_lugares, urls_provincias) — dos diccionarios con las URLs generadas.
    """    
    
    urls_lugares = {}
    urls_provincias = {}

    # URLs de lugares concretos (ciudades o landmarks): dest_type varía según el tipo de lugar
    for clave,valor in LUGARES.items():
        tipo_destino = 'landmark' if 'malagueta' in clave.lower() else 'city'
        url = f'https://www.booking.com/searchresults.es.html?ss={valor[0]}%2C+España&dest_id={valor[1]}&dest_type={tipo_destino}&checkin={FECHA_ENTRADA}&checkout={FECHA_SALIDA}&group_adults={N_ADULTOS}&no_rooms={N_HABITACIONES}&group_children={N_MENORES}'
        urls_lugares[clave] = url
    logger.info(f'Urls de lugares generadas: {len(urls_lugares)}')

    # URLs de provincias completas: dest_type=region abarca todos los municipios de la provincia
    for clave,valor in PROVINCIAS.items():
        url = f'https://www.booking.com/searchresults.es.html?ss={valor[0]}+provincia%2C+España&dest_id={valor[1]}&dest_type=region&checkin={FECHA_ENTRADA}&checkout={FECHA_SALIDA}&group_adults={N_ADULTOS}&no_rooms={N_HABITACIONES}&group_children={N_MENORES}'
        urls_provincias[valor[0]] = url
    logger.info(f'Urls de provincias generadas: {len(urls_provincias)}')

    return urls_lugares, urls_provincias

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():

    # Generamos las URLs de búsqueda para todos los lugares y provincias configurados
    urls_lugares, urls_provincias = generador_urls(FECHA_ENTRADA, FECHA_SALIDA, N_ADULTOS, N_HABITACIONES, N_MENORES, LUGARES, PROVINCIAS)
    BASE = conseguir_ruta_general_TFG()

    # Convertimos los dicts de URLs a DataFrames para guardarlos como CSV
    df_lugares = pd.DataFrame(urls_lugares.items(), columns=['localizacion','url'])
    df_provincias = pd.DataFrame(urls_provincias.items(), columns=['localizacion','url'])

    # Guardamos las URLs y las fechas en data/raw/inputs/ para que los scrapers las lean
    df_lugares.to_csv(BASE / "data" / "raw" / "inputs" / "urls_busqueda_booking_lugares.csv", index=False, sep='|')
    df_provincias.to_csv(BASE / "data" / "raw" / "inputs" / "urls_busqueda_booking_provincias.csv", index=False, sep='|')
    with open(BASE / "data" / "raw" / "inputs" / "fecha_entrada_busqueda_booking.txt", "w") as f:
        f.write(FECHA_ENTRADA)
    with open(BASE / "data" / "raw" / "inputs" / "fecha_salida_busqueda_booking.txt", "w") as f:
        f.write(FECHA_SALIDA)
    logger.info("✓ Guardado: urls_busqueda_booking_lugares.csv, urls_busqueda_booking_provincias.csv, fecha_entrada_busqueda_booking.txt, fecha_salida_busqueda_booking.txt")


if __name__ == "__main__":
    main()