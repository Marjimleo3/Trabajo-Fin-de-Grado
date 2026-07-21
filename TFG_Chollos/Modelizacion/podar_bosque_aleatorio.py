'''
podar_bosque_aleatorio.py
=========================
Genera una versión reducida del Bosque Aleatorio ganador, quedándose solo con los primeros N árboles del ensemble ya entrenado (sin reentrenar).

Motivación: el modelo completo (100 árboles) ocupa ~1,4 GB en memoria al cargarlo, muy por encima del límite (~1 GB) del plan gratuito de Streamlit Community Cloud, donde crashea la app. Podar a 20 árboles reduce la memoria a ~373 MB con una pérdida de rendimiento mínima (R² 0,8676 → 0,8634 en test), manteniendo una ventaja enorme sobre los demás modelos (árbol de decisión R²=0,59, XGBoost R²=0,51). El archivo resultante (~59 MB comprimido) cabe además sin problema en GitHub, evitando la descarga desde Hugging Face Hub.

Uso:
    python TFG_Chollos/Modelizacion/podar_bosque_aleatorio.py
'''

# =============================================================================
# IMPORTS
# =============================================================================
import copy

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from TFG_Chollos.Modelizacion.transformaciones import train_test_validation_particion
from TFG_Chollos.utils import conseguir_ruta_general_TFG, configurar_logger

# =============================================================================
# CONSTANTES
# =============================================================================
N_ARBOLES = 20

logger = configurar_logger(__name__)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    BASE = conseguir_ruta_general_TFG()

    logger.info('Cargando dataset de prueba y modelo completo...')
    db = pd.read_parquet(BASE / 'data' / 'processed' / 'modelizacion' / 'db_final_codificada.parquet')
    X = db.drop(columns=['precio'])
    y = db['precio']
    _, _, X_test, _, _, y_test = train_test_validation_particion(X, y)

    rf_completo = joblib.load(BASE / 'data' / 'models' / 'bosque_aleatorio_reg.pkl')

    rf_ligero = copy.copy(rf_completo)
    rf_ligero.estimators_ = rf_completo.estimators_[:N_ARBOLES]
    rf_ligero.n_estimators = N_ARBOLES

    y_pred = rf_ligero.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info(f'[OK] RF podado a {N_ARBOLES} árboles — R²={r2:.4f} · MAE={mae:.2f}€ · RMSE={rmse:.2f}€')

    ruta = BASE / 'data' / 'models' / 'bosque_aleatorio_reg_ligero.pkl'
    joblib.dump(rf_ligero, ruta, compress=3)
    logger.info(f'[OK] Guardado: {ruta} ({ruta.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
