"""
obtener_coordenadas_centros.py
==============================
Consulta la API de Nominatim (OpenStreetMap) para obtener las coordenadas del
centroide administrativo de cada localidad presente en los datos raw y las guarda en:

    data/raw/inputs/coordenadas_centros.csv

Ejecutar una sola vez antes de lanzar preprocessing.py.
Si ya existe el CSV, solo consulta las localidades nuevas que no estén en él.

Uso:
    python obtener_coordenadas_centros.py
"""

import glob
import time

import pandas as pd
import requests

from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG

logger = configurar_logger(__name__)

RUTA_SALIDA = 'data/raw/inputs/coordenadas_centros.csv'


def extraer_localidad(raw: pd.DataFrame) -> pd.Series:
    """
    Extrae el nombre de la localidad del campo 'direccion' (patrón '12345 Ciudad, España') con regex.
    """
    localidad = raw['direccion'].str.extract(r',\s*\d{5}\s+([^,]+),')[0]
    return localidad.fillna(raw['lugar'])


def obtener_localidades_raw(base) -> dict[str, str]:
    """ 
    Recorre todos los CSV raw del disco, llama a extraer_localidad en cada uno y construye un diccionario {localidad: provincia} con las 943 localidades únicas de todas las provincias
    """
    archivos = glob.glob(str(base / 'data' / 'raw' / 'fichas' / 'resultados_booking_*.csv'))
    localidades = {}
    for ruta in sorted(archivos):
        provincia = ruta.split('resultados_booking_')[1].replace('.csv', '')
        raw = pd.read_csv(ruta, sep='|', usecols=['lugar', 'direccion'], on_bad_lines='skip')
        raw['localidad'] = extraer_localidad(raw)
        for loc in raw['localidad'].dropna().unique():
            if loc not in localidades:
                localidades[loc] = provincia
    return localidades


def consultar_nominatim(localidad: str, provincia: str) -> tuple[float, float] | None:
    time.sleep(1)  # Nominatim: máximo 1 req/segundo
    try:
        resp = requests.get(
            'https://nominatim.openstreetmap.org/search',
            params={'q': f'{localidad}, {provincia}, Andalucía, España', 'format': 'json', 'limit': 1},
            headers={'User-Agent': 'TFG_Chollos/1.0 (mariosevillista002@gmail.com)'},
            timeout=10
        )
        resp.raise_for_status()
        res = resp.json()
        if res:
            return float(res[0]['lat']), float(res[0]['lon'])
    except Exception as e:
        logger.warning(f'Error consultando {localidad}: {e}')
    return None


def main():
    BASE = conseguir_ruta_general_TFG()
    ruta_csv = BASE / RUTA_SALIDA

    # Si ya existe el CSV, cargamos las localidades ya consultadas para no repetirlas
    if ruta_csv.exists():
        existentes = pd.read_csv(ruta_csv).set_index('localidad')[['lat_centro', 'lon_centro']].to_dict('index')
        existentes = {k: (v['lat_centro'], v['lon_centro']) for k, v in existentes.items()}
        logger.info(f'CSV existente con {len(existentes)} localidades. Solo se consultarán las nuevas.')
    else:
        existentes = {}

    # Obtenemos todas las localidades únicas de los CSV raw y filtramos las que faltan
    todas  = obtener_localidades_raw(BASE)
    nuevas = {loc: prov for loc, prov in todas.items() if loc not in existentes}
    logger.info(f'Localidades totales: {len(todas)} | Ya en CSV: {len(existentes)} | A consultar: {len(nuevas)}')

    # Consultamos Nominatim para cada localidad nueva (respetando el límite de 1 req/seg)
    resultados    = dict(existentes)
    sin_resultado = []

    for i, (localidad, provincia) in enumerate(nuevas.items(), 1):
        coords = consultar_nominatim(localidad, provincia)
        if coords:
            resultados[localidad] = coords
            logger.info(f'  [{i}/{len(nuevas)}] {localidad} ({provincia}): {coords}')
        else:
            sin_resultado.append(localidad)
            logger.warning(f'  [{i}/{len(nuevas)}] Sin resultado para: {localidad} ({provincia})')

    # Guardamos el CSV completo (existentes + nuevas) ordenado por localidad
    filas = [{'localidad': loc, 'lat_centro': coords[0], 'lon_centro': coords[1]}
             for loc, coords in resultados.items()]
    pd.DataFrame(filas).sort_values('localidad').to_csv(ruta_csv, index=False)
    logger.info(f'[OK] Guardadas {len(filas)} localidades en {ruta_csv}')

    if sin_resultado:
        logger.warning(f'[WARN] {len(sin_resultado)} localidades sin coordenadas: {sin_resultado}')


if __name__ == '__main__':
    main()
