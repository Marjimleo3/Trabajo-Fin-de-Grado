"""
encoding.py
====================
Codifica la base de datos analítica para su uso en modelos de ML.
Lee db_final_analisis.parquet y genera db_final_codificada.parquet.

Para variables categóricas:
    One-Hot Encoding  → crea columnas binarias (0/1) por cada categoría.
    Label Encoding    → asigna un número entero a cada categoría.
    Ordinal Encoding  → como label encoding pero respetando un orden lógico.
    Target Encoding   → reemplaza la categoría por la media del target.
Para variables de fecha/hora:
    Extraer componentes numéricos: día, mes, hora, día de la semana, etc.

Uso:
    python Cleaning/encoding.py
"""

# =============================================================================
# IMPORTS
# =============================================================================
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
def encoding(db_final: pd.DataFrame) -> pd.DataFrame:

    # Eliminamos columnas identificativas que no aportan valor predictivo al modelo
    db_final = db_final.drop(columns=['titulo', 'codigo_postal', 'url_estancia', 'fecha_extraccion', 'latitud', 'longitud', 'latitud_centro', 'longitud_centro'])

    # One-Hot Encoding de 'tipo': tipo_Hotel=1 si es hotel, 0 si no. Se elimina tipo_Otro por redundancia.
    db_final = pd.get_dummies(db_final, columns=['tipo'], drop_first=False)
    db_final = db_final.drop(columns=['tipo_Otro'])

    # Label Encoding de localidad y provincia: asigna un entero a cada categoría. Se eligió Label Encoding (en lugar de OHE) porque el alto número de categorías habría generado una dimensionalidad excesiva con One-Hot Encoding.
    le_localidad = LabelEncoder()
    le_provincia = LabelEncoder()
    db_final['localidad'] = le_localidad.fit_transform(db_final['localidad'])
    db_final['provincia'] = le_provincia.fit_transform(db_final['provincia'])

    # Descomposición de 'fecha_disponible' en mes y día como variables numéricas, luego la eliminamos
    db_final['mes_disponible'] = db_final['fecha_disponible'].dt.month
    db_final['dia_disponible'] = db_final['fecha_disponible'].dt.day
    db_final = db_final.drop(columns=['fecha_disponible'])

    # Guardamos los encoders y la lista de columnas para poder replicar el encoding en producción
    models_dir = BASE / 'data' / 'models'
    joblib.dump(le_localidad, models_dir / 'le_localidad.pkl')
    joblib.dump(le_provincia, models_dir / 'le_provincia.pkl')
    columnas_modelo = db_final.drop(columns=['precio']).columns.tolist()
    joblib.dump(columnas_modelo, models_dir / 'columnas_modelo.pkl')

    return db_final


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    db_analisis = pd.read_parquet(BASE / "data" / "processed" / "analisis" / "db_final_analisis.parquet")

    db_codificada = encoding(db_analisis)
    db_codificada.to_parquet(BASE / "data" / "processed" / "modelizacion" / "db_final_codificada.parquet", index=False)
    logger.info('[OK] Dataset completo codificado guardado correctamente')
    logger.info('[OK] LabelEncoders y columnas del modelo guardados en data/models/')


if __name__ == "__main__":
    main()
