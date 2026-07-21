"""
Modelizacion.py
===============
Entrenamiento, selección y evaluación de los modelos de regresión para
predecir el precio justo de un alojamiento.

Estrategia de validación:
    Partición holdout 70/15/15 combinada con validación cruzada K-Fold (K=3)
    mediante GridSearchCV para el ajuste de hiperparámetros.

    X_train  →  entrenar cada modelo
    X_val    →  ajustar hiperparámetros + elegir el mejor modelo   ← tomas decisiones
    X_test   →  evaluar el modelo elegido y reportar               ← no tomas decisiones

Modelos entrenados:
    - Regresión Lineal Múltiple
    - Árbol de Decisión
    - Bosque Aleatorio (modelo principal, cargado desde disco)
    - XGBoost (Boosting)

Uso:
    python Modelizacion/Modelizacion.py
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG
from TFG_Chollos.Modelizacion.transformaciones import (
    train_test_validation_particion,
    estandarizar_datos,
)
from TFG_Chollos.Modelizacion.categorizacion import crear_etiqueta_chollo

# =============================================================================
# CONSTANTES
# =============================================================================
BASE   = conseguir_ruta_general_TFG()
logger = configurar_logger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def _log_metricas(y_real, y_pred, etiqueta: str, units: str = '€'):
    """Calcula y loguea R², MAE y RMSE."""
    r2   = r2_score(y_real, y_pred)
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    logger.info(f'{etiqueta} R²={r2:.4f} | MAE={mae:.2f}{units} | RMSE={rmse:.2f}{units}')
    return r2


# =============================================================================
# MODELOS DE REGRESIÓN
# =============================================================================

def crear_regresion_lineal(conjunto_ent_est: pd.DataFrame, conjunto_val_est: pd.DataFrame,
                            target_ent_est: pd.Series, target_val_est: pd.Series,
                            variable_representar: str,
                            X_plot: pd.DataFrame = None, y_plot: pd.Series = None):
    """
    Entrena una Regresión Lineal Múltiple sobre datos estandarizados.
    Las métricas de validación se reportan en unidades z-score porque tanto X como y
    están estandarizados; la conversión a € se realiza en main() con inverse_transform.
    Genera un gráfico de dispersión en unidades originales si se pasan X_plot e y_plot,
    o en escala estandarizada en caso contrario.
    """
    logger.info('Creando "Regresión Lineal Múltiple"')

    regresion = linear_model.LinearRegression()
    regresion.fit(conjunto_ent_est, target_ent_est)
    logger.info('[OK] Modelo entrenado')

    y_pred_val = regresion.predict(conjunto_val_est)
    _log_metricas(target_val_est, y_pred_val, '[VAL - Regresión Lineal]', units=' (z-score)')

    ruta_modelo = BASE / 'data' / 'models' / 'regresion_lineal_reg.pkl'
    joblib.dump(regresion, ruta_modelo)
    logger.info(f'[OK] Modelo guardado: {ruta_modelo}')

    # Gráfico: dispersión en unidades originales si se pasan X_plot/y_plot, estandarizadas si no
    X_graf = X_plot if X_plot is not None else conjunto_ent_est
    y_graf = y_plot if y_plot is not None else target_ent_est
    xlabel = f'{variable_representar} (m²)' if X_plot is not None else f'{variable_representar} (z-score)'
    ylabel = 'precio (€)' if X_plot is not None else 'precio (z-score)'
    sns.scatterplot(x=X_graf[variable_representar], y=y_graf, alpha=0.3, s=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Relación entre {variable_representar} y precio')

    ruta = BASE / 'images' / f'regresion_lineal_{variable_representar}.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f'[OK] Gráfico guardado: {ruta}')

    return regresion


def crear_arbol_decision(conjunto_ent: pd.DataFrame, conjunto_val: pd.DataFrame,
                          target_ent: pd.Series, target_val: pd.Series):
    """
    Entrena un Árbol de Decisión con búsqueda en rejilla de max_depth.
    No requiere escalado: solo compara umbrales, la magnitud no importa.
    Métricas de validación en €.
    Genera un gráfico del árbol con profundidad 3 para visualizar las decisiones principales.
    """
    logger.info('Creando "Árbol de Decisión"')

    param_grid = {'max_depth': [3, 5, 10, 15]}

    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(conjunto_ent, target_ent)
    mejor_arbol = grid_search.best_estimator_
    logger.info(f'[OK] Modelo entrenado | Mejores hiperparámetros: {grid_search.best_params_}')

    y_pred_val = mejor_arbol.predict(conjunto_val)
    _log_metricas(target_val, y_pred_val, '[VAL - Árbol de Decisión]')

    ruta_modelo = BASE / 'data' / 'models' / 'arbol_decision_reg.pkl'
    joblib.dump(mejor_arbol, ruta_modelo)
    logger.info(f'[OK] Modelo guardado: {ruta_modelo}')

    # Visualizamos las 3 primeras capas del árbol (las decisiones más importantes)
    plt.figure(figsize=(20, 10))
    plot_tree(
        decision_tree=mejor_arbol,
        filled=True,
        max_depth=3,
        fontsize=9,
        feature_names=conjunto_ent.columns.to_list(),
    )
    ruta = BASE / 'images' / 'arbol_decision.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f'[OK] Gráfico guardado: {ruta}')

    return mejor_arbol


def crear_bosque_aleatorio(conjunto_ent: pd.DataFrame, conjunto_val: pd.DataFrame,
                            target_ent: pd.Series, target_val: pd.Series):
    """
    Entrena un Random Forest con búsqueda en rejilla de n_estimators y max_depth.
    Técnica Bagging: combina múltiples árboles con subconjuntos aleatorios de datos y features.
    Métricas de validación en €.
    No requiere escalado.
    """
    logger.info('Creando "Bosque Aleatorio"')

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth':    [10, 20],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(conjunto_ent, target_ent)
    mejor_bosque = grid_search.best_estimator_
    logger.info(f'[OK] Modelo entrenado | Mejores hiperparámetros: {grid_search.best_params_}')

    y_pred_val = mejor_bosque.predict(conjunto_val)
    _log_metricas(target_val, y_pred_val, '[VAL - Bosque Aleatorio]')

    ruta_modelo = BASE / 'data' / 'models' / 'bosque_aleatorio_reg.pkl'
    joblib.dump(mejor_bosque, ruta_modelo)
    logger.info(f'[OK] Modelo guardado: {ruta_modelo}')

    return mejor_bosque


def crear_boosting(conjunto_ent: pd.DataFrame, conjunto_val: pd.DataFrame,
                   target_ent: pd.Series, target_val: pd.Series):
    """
    Entrena un XGBoost con búsqueda en rejilla de n_estimators, learning_rate y max_depth.
    Gradient Boosting: construye árboles secuencialmente corrigiendo los errores del anterior.
    Métricas de validación en €.
    No requiere escalado.
    """
    logger.info('Creando "XGBoost"')

    param_grid = {
        'n_estimators':  [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth':     [3, 5],
    }

    grid_search = GridSearchCV(
        estimator=XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(conjunto_ent, target_ent)
    mejor_boosting = grid_search.best_estimator_
    logger.info(f'[OK] Modelo entrenado | Mejores hiperparámetros: {grid_search.best_params_}')

    y_pred_val = mejor_boosting.predict(conjunto_val)
    _log_metricas(target_val, y_pred_val, '[VAL - XGBoost]')

    ruta_modelo = BASE / 'data' / 'models' / 'boosting_reg.pkl'
    joblib.dump(mejor_boosting, ruta_modelo)
    logger.info(f'[OK] Modelo guardado: {ruta_modelo}')

    return mejor_boosting


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main():

    db = pd.read_parquet(
        BASE / 'data' / 'processed' / 'modelizacion' / 'db_final_codificada.parquet'
    )
    X = db.drop(columns=['precio'])
    y = db['precio']

    # 1. PARTICIÓN 70 / 15 / 15
    # -------------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_particion(X, y)

    # 2. ESTANDARIZACIÓN (solo para Regresión Lineal)
    # -------------------------------------------------------------------------
    X_train_est, X_val_est, X_test_est, y_train_est, y_val_est, y_test_est, scaler_X, scaler_y = \
        estandarizar_datos(X_train, X_val, X_test, y_train, y_val, y_test)

    # 3. ENTRENAMIENTO DE MODELOS
    # Las funciones loguean métricas de validación inmediatamente tras el entrenamiento. Para LR las métricas son en z-score; el resto en €.
    # -------------------------------------------------------------------------
    regresion = crear_regresion_lineal(X_train_est, X_val_est, y_train_est, y_val_est, 'tamaño_habitacion', X_plot=X_train, y_plot=y_train)
    arbol     = crear_arbol_decision(X_train, X_val, y_train, y_val)
    bosque    = crear_bosque_aleatorio(X_train, X_val, y_train, y_val)
    boosting  = crear_boosting(X_train, X_val, y_train, y_val)

    # 4. SELECCIÓN DEL MEJOR MODELO + EVALUACIÓN EN TEST
    # Todas las métricas en € (inverse_transform para LR) para que sean comparables.
    # La selección se basa en R² val; el test se calcula pero no influye en la decisión.
    # -------------------------------------------------------------------------
    modelos_regresion = [
        ('Regresión Lineal',  regresion, X_train_est, X_val_est, X_test_est, y_train_est, y_val_est, y_test_est),
        ('Árbol de Decisión', arbol,     X_train,     X_val,     X_test,     y_train,     y_val,     y_test),
        ('Bosque Aleatorio',  bosque,    X_train,     X_val,     X_test,     y_train,     y_val,     y_test),
        ('Boosting',          boosting,  X_train,     X_val,     X_test,     y_train,     y_val,     y_test),
    ]

    mejor_score = None
    mejor_nombre = mejor_modelo_reg = None
    mejor_X_train = mejor_X_val = mejor_X_test = None
    mejor_y_train = mejor_y_val = mejor_y_test = None

    for nombre, modelo, X_tr, X_va, X_te, y_tr, y_va, y_te in modelos_regresion:
        y_pred_va = modelo.predict(X_va)
        y_pred_te = modelo.predict(X_te)

        # LR: inverse_transform para obtener MAE/RMSE en € comparables con el resto
        if nombre == 'Regresión Lineal':
            y_pred_va_eur = scaler_y.inverse_transform(y_pred_va.reshape(-1, 1)).ravel()
            y_pred_te_eur = scaler_y.inverse_transform(y_pred_te.reshape(-1, 1)).ravel()
            y_va_eur = y_val.values
            y_te_eur = y_test.values
        else:
            y_pred_va_eur, y_pred_te_eur = y_pred_va, y_pred_te
            y_va_eur = y_va.values
            y_te_eur = y_te.values

        r2_va = _log_metricas(y_va_eur, y_pred_va_eur, f'[VAL  - {nombre}]')
        _log_metricas(y_te_eur, y_pred_te_eur, f'[TEST - {nombre}]')

        if mejor_score is None or r2_va > mejor_score:
            mejor_score  = r2_va
            mejor_nombre, mejor_modelo_reg = nombre, modelo
            mejor_X_train, mejor_X_val, mejor_X_test = X_tr, X_va, X_te
            mejor_y_train, mejor_y_val, mejor_y_test = y_tr, y_va, y_te

    logger.info(f'[OK] Mejor modelo de regresión: {mejor_nombre} (R² val={mejor_score:.4f})')

    # Énfasis especial en Bosque Aleatorio: importancia de variables
    if mejor_nombre == 'Bosque Aleatorio':
        importancias = pd.Series(
            mejor_modelo_reg.feature_importances_,
            index=mejor_X_train.columns,
        ).sort_values(ascending=False)
        logger.info('Top 10 variables más importantes (Bosque Aleatorio):')
        for feat, imp in importancias.head(10).items():
            logger.info(f'  {feat:35s} {imp:.4f}')

    # 5. ETIQUETADO DE CHOLLOS con el mejor modelo
    # -------------------------------------------------------------------------
    y_pred_train = mejor_modelo_reg.predict(mejor_X_train)
    chollo_train = crear_etiqueta_chollo(mejor_y_train, pd.Series(y_pred_train, index=mejor_y_train.index))
    logger.info(f'Distribución categorías train: {chollo_train.value_counts().to_dict()}')

    # 6. PREDICCIONES TEST del modelo ganador (ya evaluadas en el paso 4)
    # -------------------------------------------------------------------------
    y_pred_test_reg = mejor_modelo_reg.predict(mejor_X_test)

    # 7. GUARDAR SCALERS del modelo ganador (solo si es Regresión Lineal)
    # -------------------------------------------------------------------------
    if mejor_nombre == 'Regresión Lineal':
        modelos_dir = BASE / 'data' / 'models'
        joblib.dump(scaler_X, modelos_dir / 'scaler_X_regresion.pkl')
        joblib.dump(scaler_y, modelos_dir / 'scaler_y_regresion.pkl')
        logger.info(f'[OK] Scalers guardados en {modelos_dir}')


if __name__ == '__main__':
    main()
