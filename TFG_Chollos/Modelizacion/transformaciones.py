"""
transformaciones.py
===================
Partición del dataset y transformación de escala de los datos
para los modelos de regresión del pipeline de modelización.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PARTICIÓN
# =============================================================================

def train_test_validation_particion(features: pd.DataFrame,
                                     target: pd.Series) -> tuple:
    """
    Divide el dataset en tres conjuntos con proporción 70 / 15 / 15.

    Estrategia:
        1. Partición 70-30 → train y temporal
        2. Partición 50-50 del temporal → val y test

    Devuelve
    --------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Primera partición: 70% train, 30% temporal
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, target, train_size=0.7, random_state=42
    )

    # Segunda partición: 50% validación, 50% test (sobre el 30% temporal)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# ESTANDARIZACIÓN
# =============================================================================

def estandarizar_datos(conjunto_ent: pd.DataFrame, conjunto_val: pd.DataFrame,
                        conjunto_test: pd.DataFrame, target_ent: pd.Series,
                        target_val: pd.Series, target_test: pd.Series) -> tuple:
    """
    Aplica Z-score scaling: cada columna queda con media 0 y desviación típica 1.
    Fórmula: z = (x - μ) / σ

    El scaler se ajusta SOLO sobre train y se aplica sobre val y test,
    para no filtrar información del futuro al modelo.

    Se usa para la Regresión Lineal (sensible a la magnitud de las variables).

    Devuelve
    --------
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

    Transformación inversa: scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # TRAIN: el scaler aprende la media y std, luego transforma
    X_train = scaler_X.fit_transform(conjunto_ent)
    # y_train es 1D → reshape a 2D para el scaler, luego ravel() vuelve a 1D
    y_train = scaler_y.fit_transform(target_ent.values.reshape(-1, 1)).ravel()

    # VAL y TEST: solo transforma con los parámetros aprendidos en train
    X_val  = scaler_X.transform(conjunto_val)
    X_test = scaler_X.transform(conjunto_test)
    y_val  = scaler_y.transform(target_val.values.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(target_test.values.reshape(-1, 1)).ravel()

    # Convertimos los arrays de numpy de vuelta a DataFrames y Series
    X_train, X_val, X_test = [
        pd.DataFrame(a, columns=conjunto_ent.columns)
        for a in [X_train, X_val, X_test]
    ]
    y_train, y_val, y_test = [
        pd.Series(a, name=target_ent.name)
        for a in [y_train, y_val, y_test]
    ]

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y
