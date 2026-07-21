"""
categorizacion.py
=================
Etiquetado de categorías de chollo a partir del ratio precio_real / precio_predicho.

Módulo deliberadamente ligero (solo numpy/pandas): lo usan tanto Modelizacion.py
(entrenamiento) como las apps de Streamlit (App/App_Cloud), y estas últimas no
necesitan arrastrar matplotlib/seaborn/sklearn/xgboost solo para categorizar.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# ETIQUETADO DE CHOLLOS
# =============================================================================

def crear_etiqueta_chollo(y_real: pd.Series, y_predicho: pd.Series,
                          umbral_hiper: float = 0.75, umbral_super: float = 0.85,
                          umbral_chollo: float = 0.97, umbral_inflado: float = 1.03) -> pd.Series:
    """
    Clasifica cada alojamiento según el ratio precio_real / precio_predicho:

        hiper_chollo : 4  →  ratio <= umbral_hiper
        super_chollo : 3  →  umbral_hiper < ratio < umbral_super
        chollo       : 2  →  umbral_super <= ratio < umbral_chollo
        normal       : 1  →  umbral_chollo <= ratio <= umbral_inflado
        inflado      : 0  →  ratio > umbral_inflado

    Los valores por defecto (0.75, 0.85, 0.97, 1.03) son los umbrales canónicos
    usados para etiquetar el dataset de entrenamiento; no cambiarlos aquí.
    Las apps pueden pasar umbrales distintos para el control de exigencia del usuario.
    """
    ratio = y_real / y_predicho

    condiciones = [
        ratio <= umbral_hiper,
        (ratio > umbral_hiper)  & (ratio < umbral_super),
        (ratio >= umbral_super) & (ratio < umbral_chollo),
        (ratio >= umbral_chollo) & (ratio <= umbral_inflado),
        ratio > umbral_inflado,
    ]
    etiquetas = [4, 3, 2, 1, 0]

    # np.select asigna a cada elemento la etiqueta de la primera condición True
    return pd.Series(
        np.select(condiciones, etiquetas),
        index=y_real.index,
        name='categoria',
    )
