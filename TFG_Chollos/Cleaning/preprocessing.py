"""
preprocessing.py
====================
Limpia y estructura la base de datos proveniente de scraping

Dependencias:
    - Python >= 3.10
    - pandas >= 3.0.1

Requisitos:
    uv
    uv add pandas --active --link-mode=copy

Uso:
    python preprocessing.py --input data_Booking/resultados/resultados_booking_{provincia}.csv  --output data_Booking/final/db_final_{provincia}.parquet
"""

# =============================================================================
# IMPORTS
# =============================================================================
#Librerías estándar (vienen incluidas con Python):
import ast
import json
import math

#Librerías de terceros (es necesario instalarlas):
import numpy as np
import pandas as pd

#Módulos propios del proyecto
from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()

# =============================================================================
# CONFIGURACIÓN DEL LOGGER
# =============================================================================
logger = configurar_logger(__name__)

# =============================================================================
# FUNCIONES
# =============================================================================
def extraer_servicios_influyentes(ficha:pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las listas de servicios del scraper en variables binarias (True/False).
    Recorre los servicios generales del alojamiento y los de la habitación
    buscando palabras clave para cada amenity relevante para el modelo.
    """
    lista = []

    for _, fila in ficha.iterrows():
        # ast.literal_eval convierte el string con formato de lista Python en una lista real
        servicios = [serv.lower().strip('"') for serv in ast.literal_eval(fila['servicios'])]
        servicios_habitacion = [serv.lower().strip('"') for serv in ast.literal_eval(fila['servicios_habitacion'])]
        
        parking = parking_gratis = gimnasio = restaurante = piscina = piscina_interior = piscina_infinita = aire = calefaccion = vistas = terraza = baño_privado = False
        #Incluir tamaño_habitacion cuando esté listo

        for servicio,servicio_habitacion in zip(servicios,servicios_habitacion):
            if 'parking' in servicio and ('parking fuera del alojamiento' not in servicio and 'parking en la calle' not in servicio):
                parking = servicio
            if 'parking gratis' in servicio or 'free parking' in servicio:
                parking_gratis = servicio
            if ('gimnasio' in servicio or 'gym' in servicio) and 'taquillas en el gimnasio / spa' not in servicio:
                gimnasio = servicio
            if 'restaurante' in servicio or 'restaurant' in servicio:
                restaurante = servicio
            if ('piscina' in servicio or 'pool' in servicio) and 'vistas a la piscina' not in servicio:
                piscina = servicio
            if 'piscina interior' in servicio or 'cubierta' in servicio:
                piscina_interior = servicio
            if 'piscina infinita' in servicio or 'infinity pool' in servicio:
                piscina_infinita = servicio

            if ('aire' in servicio_habitacion or 'air' in servicio_habitacion) and ('aire libre' not in servicio_habitacion and 'stairs' not in servicio_habitacion and 'hair' not in servicio_habitacion and 'chair' not in servicio_habitacion and 'purifier' not in servicio_habitacion):
                aire = servicio_habitacion
            if 'calefacción' in servicio_habitacion or 'calefaccion' in servicio_habitacion or 'heat' in servicio_habitacion:
                calefaccion = servicio_habitacion
            if ('vista' in servicio_habitacion or 'views' in servicio_habitacion or 'scenary' in servicio_habitacion) and ('pay-per-view channels' not in servicio_habitacion and 'piscina con vistas' not in servicio_habitacion):
                vistas = servicio_habitacion
            if 'terraza' in servicio_habitacion or 'terrace' in servicio_habitacion or 'deck' in servicio_habitacion or 'balcón' in servicio_habitacion or 'balcon' in servicio_habitacion:
                terraza = servicio_habitacion
            if ('baño' in servicio_habitacion or 'bath' in servicio_habitacion) and ('shared bathroom' not in servicio_habitacion and 'bath-robe' not in servicio_habitacion):
                baño_privado = servicio_habitacion
            # if ('m²' in servicio_habitacion):
            #     tamaño_habitacion = servicio_habitacion


        lista.append({            
            'Parking': parking != False,
            'Parking_gratis' : parking_gratis != False,             
            'Gimnasio': gimnasio != False,             
            'Restaurante': restaurante != False,             
            'Piscina': piscina != False,
            'Piscina_interior': piscina_interior != False,
            'Piscina_infinita': piscina_infinita != False,
            'Aire': aire != False,
            'Calefaccion': calefaccion != False,
            'Vistas': vistas != False,
            'Terraza': terraza != False,
            'Baño_privado': baño_privado != False,
            # 'Tamaño_habitacion' : tamaño_habitacion != False,
            'url_estancia' : fila['url_estancia']
            })

    return pd.DataFrame(lista)



def extraer_fecha_precios_disponibles(ficha:pd.DataFrame) -> pd.DataFrame:
    """
    Expande el calendario JSON de cada alojamiento en una fila por día disponible.
    Solo incluye días con disponible=True, generando el par (fecha, precio) por alojamiento.
    """
    lista_precios = []
    for _, fila in ficha.iterrows():
        # json.loads convierte el string JSON del calendario en una lista de dicts Python
        calendario_limpio = json.loads(fila['calendario'])

        for registro in calendario_limpio:
            if registro['disponible'] == True:
                lista_precios.append({
                    'fecha_disponible' : registro['fecha'],
                    'precio' : registro['precio'],
                    'url_estancia' : fila['url_estancia']
                })

    return pd.DataFrame(lista_precios)



def limpiar_room_size(room_size:pd.DataFrame) -> pd.DataFrame:
    room_size = room_size.drop_duplicates(subset='url_estancia')
    return room_size



def añadir_columnas_fechas(db_semifinal:pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna 'fecha_extraccion' según la provincia y calcula 'dias_restantes'
    como la diferencia en días entre la fecha de disponibilidad y la de extracción.
    """
    mapa_fechas = {
        'Sevilla':  '2026-05-13',
        'Cádiz':    '2026-05-14',
        'Huelva':   '2026-05-14',
        'Jaén':     '2026-05-14',
        'Granada':  '2026-05-15',
        'Almería':  '2026-05-15',
        'Córdoba':  '2026-05-15',
        'Málaga':   '2026-05-16'
        }

    db_semifinal['fecha_extraccion'] = db_semifinal['lugar'].map(mapa_fechas)

    db_semifinal['fecha_extraccion'] = pd.to_datetime(db_semifinal['fecha_extraccion'])
    db_semifinal['fecha_disponible'] = pd.to_datetime(db_semifinal['fecha_disponible'])
    db_semifinal['dias_restantes'] = db_semifinal['fecha_disponible'] - db_semifinal['fecha_extraccion']
    db_semifinal['dias_restantes'] = db_semifinal['dias_restantes'].dt.days

    return db_semifinal



def reordenar_df(db_semifinal:pd.DataFrame) -> pd.DataFrame:
    col = db_semifinal.pop('fecha_extraccion')
    db_semifinal.insert(24, 'fecha_extraccion', col)
    col1 = db_semifinal.pop('dias_restantes')
    db_semifinal.insert(26, 'dias_restantes', col1)

    return db_semifinal



def extraer_localidad(raw: pd.DataFrame) -> pd.Series:
    """
    Extrae el nombre de la localidad del campo 'direccion' (patrón '12345 Ciudad, España') con regex.
    """
    localidad = raw['direccion'].str.extract(r',\s*\d{5}\s+([^,]+),')[0]
    return localidad



def cargar_coords_centros(base) -> dict[str, tuple[float, float]]:
    ruta = base / 'data' / 'raw' / 'inputs' / 'coordenadas_centros.csv'
    if not ruta.exists():
        raise FileNotFoundError(
            f'No se encontró {ruta}. Ejecuta primero Cleaning/obtener_coordenadas_centros.py'
        )
    df = pd.read_csv(ruta)
    return {row['localidad']: (row['lat_centro'], row['lon_centro']) for _, row in df.iterrows()}



def haversine_km_vec(lats: pd.Series, lons: pd.Series, lat2: float, lon2: float) -> pd.Series:
    '''
    Calcula la distancia vectorizada con numpy (rápida para 100k+ filas) a través de las latitudes y las longitudes
    '''
    R = 6371.0
    dlat = np.radians(lat2 - lats)
    dlon = np.radians(lon2 - lons)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lats)) * math.cos(math.radians(lat2)) * np.sin(dlon / 2) ** 2
    return (R * 2 * np.arcsin(np.sqrt(a))).round(3)



def añadir_distancia_centro(df: pd.DataFrame, coords_centros: dict) -> pd.DataFrame:
    distancias  = pd.Series(np.nan, index=df.index, dtype='float32')
    lat_centros = pd.Series(np.nan, index=df.index, dtype='float32')
    lon_centros = pd.Series(np.nan, index=df.index, dtype='float32')

    for loc, grupo in df.groupby('localidad', observed=True):
        coords = coords_centros.get(loc)
        if coords is not None:
            lat_c, lon_c = coords
            distancias.loc[grupo.index]  = haversine_km_vec(
                grupo['latitud'], grupo['longitud'], lat_c, lon_c
            ).astype('float32')
            lat_centros.loc[grupo.index] = np.float32(lat_c)
            lon_centros.loc[grupo.index] = np.float32(lon_c)
        else:
            logger.warning(f'Sin coordenadas para "{loc}", distancia será NaN')

    idx = df.columns.get_loc('longitud') + 1
    df.insert(idx,     'distancia_centro_km', distancias)
    df.insert(idx + 1, 'latitud_centro',   lat_centros)
    df.insert(idx + 2, 'longitud_centro',  lon_centros)
    return df



def limpiar_db_final(db_final:pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las transformaciones de limpieza al dataset ensamblado:
    tipado, renombrado, tratamiento de nulos, nuevas variables y reordenación de columnas.
    """
    #Transformaciones necesarias para el tipado:
    db_final['valoracion_clientes'] = db_final['valoracion_clientes'].str.replace(',','.')
    db_final['codigo_postal'] = db_final['codigo_postal'].astype(str).str.extract(r'(\d{5})')[0]  #Esto busca el primer grupo de 5 dígitos y descarta el resto. En pandas, para aplicar métodos de texto a una Serie tienes que usar el accesor .str primero

    #Renombramiento de columnas
    db_final = db_final.rename(columns={'lugar': 'provincia'})   #Cambiamos el nombre de lugar a provincia
    db_final = db_final.rename(columns={'room_size_m2':'tamaño_habitacion'})

    #Reemplazo de valores (Eliminar en posteriores Scrapings):
    db_final['tipo'] = db_final['tipo'].replace('Apartamento/Casa/Estudio', 'Otro')

    #Procesamiento de valores nulos:
    ###Eliminación registros:
    cols_obligatorias = ['latitud', 'longitud', 'latitud_centro', 'longitud_centro', 'distancia_centro_km']   #Con latitud_centro, longitud_centro y dist_centro de pierde el 2.2% total (muy asumible), también se incluyen ahí los NaN de localidad, porque sin localidad no se puede calcular las variables que acabo de mencionar
    db_final = db_final.dropna(subset=cols_obligatorias)
    db_final = db_final[db_final['localidad'] != '9']   #Hay una localidad que se llama '9'

    ###Reemplazo de valores nulos:
    nuevos_valores = {
        'codigo_postal':-1,                                                             #Nan: valor desconocido
        'tipo':'Otro',                                              #Nan: lo metemos en el saco grande 'Apartamento/Casa/Estudio
        'estrellas':1,                                                                  #Nan: no tiene estrellas
        'n_valoraciones':0,                                                             #Nan: no tiene valoraciones
        'valoracion_clientes':db_final['valoracion_clientes'].astype(float).mean().round(1),    #Nan: rellenamos con la media de cada una de las provincias, para no sesgar el modelo hacia valoraciones extremas
        'tamaño_habitacion':db_final['tamaño_habitacion'].median()}                    #Nan: la mediana es más robusta que la media ante suites con m² extremos
    db_final = db_final.fillna(nuevos_valores)

    #Tipado de datos:
    db_final['fecha_disponible'] = pd.to_datetime(db_final['fecha_disponible'])
    bool_cols = db_final.select_dtypes(include='bool').columns  #Seleccionamos las columnas bool
    db_final[bool_cols] = db_final[bool_cols].astype('int8')
    db_final = db_final.astype({
        'provincia':'category', 
        'localidad':'category', 
        'codigo_postal':'int32',
        'latitud_centro':'float64',
        'longitud_centro':'float64',
        'tipo':'category', 
        'estrellas': 'int8',
        'valoracion_clientes':'float32',
        'n_valoraciones':'int64', 
        'tamaño_habitacion':'int16', 
        'dias_restantes':'int16', 
        'precio':'int32'})

    #Nuevas variables:
    db_final['es_finde'] = db_final['fecha_disponible'].dt.dayofweek.isin([4, 5]).astype('int8')
    db_final['es_domingo'] = (db_final['fecha_disponible'].dt.dayofweek == 6).astype('int8')

    #Reordenamos la columna 'precio':
    precio = db_final.pop('precio')
    db_final.insert(len(db_final.columns), 'precio', precio)

    return db_final



# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():

    # Cargamos la lista de provincias y los tamaños de habitación extraídos por patch_room_size.py
    provincias = pd.read_csv( BASE / "data" / "raw" / "inputs" / "urls_busqueda_booking_provincias.csv", sep="|" )
    tamaño_habitacion = pd.read_csv( BASE / "data" / "raw" / "fichas" / "room_sizes.csv", sep="|")
    tamaño_habitacion = limpiar_room_size(tamaño_habitacion)
    coords_centros = cargar_coords_centros(BASE)

    dfs_finales = []

    for provincia in provincias.iloc[:,0]:
        # Cargamos el CSV raw del scraper y extraemos la localidad a partir de la dirección
        raw = pd.read_csv( BASE / "data" / "raw" / "fichas" / f"resultados_booking_{provincia}.csv", sep="|")
        raw['localidad'] = extraer_localidad(raw)

        # Extraemos los servicios binarios y los precios por fecha disponible
        servicios_generales = extraer_servicios_influyentes(raw)
        servicios_generales.to_csv(BASE / "data" / "processed" / "servicios_binarios" / f"servicios_generales_binarios_{provincia}.csv", index=False, sep="|")

        precios_disponibles = extraer_fecha_precios_disponibles(raw)
        precios_disponibles.to_csv(BASE / "data" / "processed" / "precios" / f"precios_disponibles_{provincia}.csv", index=False, sep="|")

        # Ensamblamos el dataset provincial uniendo raw, servicios, tamaño y precios
        raw_limpio = raw[['lugar','localidad','titulo','codigo_postal','latitud','longitud','tipo','estrellas','valoracion_clientes','n_valoraciones','url_estancia']]
        df_1 = raw_limpio.merge(servicios_generales)
        df_2 = df_1.merge(tamaño_habitacion[['url_estancia','room_size_m2']])
        df_3 = df_2.merge(precios_disponibles)

        # Aplicamos las transformaciones de fechas, reordenación, distancia y limpieza final
        df_4 = añadir_columnas_fechas(df_3)
        df_5 = reordenar_df(df_4)
        df_5 = añadir_distancia_centro(df_5, coords_centros)
        df_6 = limpiar_db_final(df_5)

        logger.info(f'Ciclo de {provincia} completado')
        dfs_finales.append(df_6)

    # Concatenamos todos los datasets provinciales y guardamos el parquet final
    db_completa = pd.concat(dfs_finales, ignore_index=True)
    db_completa.to_parquet(BASE / "data" / "processed" / "final" / "db_final.parquet", index=False)
    logger.info('[OK] Dataset completo guardado correctamente')

    

if __name__ == "__main__":
    main()