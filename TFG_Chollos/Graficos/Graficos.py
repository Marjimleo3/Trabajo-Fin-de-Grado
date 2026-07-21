"""
Graficos.py
===========
Genera la figura de distribución de categorías de chollo sobre el conjunto de
prueba usando las predicciones del modelo ganador (Bosque Aleatorio).

La partición utiliza la misma semilla que Modelizacion.py (random_state=42),
por lo que el conjunto de prueba es idéntico al empleado en el entrenamiento.

Uso:
    python Graficos/Graficos.py
"""

# =============================================================================
# IMPORTS
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG
from TFG_Chollos.Modelizacion.categorizacion import crear_etiqueta_chollo
from TFG_Chollos.Modelizacion.transformaciones import train_test_validation_particion

# =============================================================================
# CONSTANTES
# =============================================================================
BASE   = conseguir_ruta_general_TFG()
logger = configurar_logger(__name__)

ETIQUETAS = {0: 'Inflado', 1: 'Normal', 2: 'Chollo', 3: 'Super-Chollo', 4: 'Hiper-Chollo'}
COLORES   = ['#d62728', '#aec7e8', '#2ca02c', '#ff7f0e', '#1f77b4']


# =============================================================================
# FUNCIONES
# =============================================================================

def grafico_distribucion_chollo_test():
    """
    Genera un gráfico de pastel con la distribución de categorías de chollo
    sobre el conjunto de prueba y lo guarda como distribucion_categorias_test.png.
    """
    logger.info('Cargando datos y modelo...')
    db = pd.read_parquet(
        BASE / 'data' / 'processed' / 'modelizacion' / 'db_final_codificada.parquet'
    )
    X = db.drop(columns=['precio'])
    y = db['precio']
    _, _, X_test, _, _, y_test = train_test_validation_particion(X, y)

    modelo = joblib.load(BASE / 'data' / 'models' / 'bosque_aleatorio_reg.pkl')
    logger.info('[OK] Modelo cargado')

    y_pred = pd.Series(modelo.predict(X_test), index=y_test.index)
    categorias = crear_etiqueta_chollo(y_test, y_pred)
    conteos = categorias.value_counts().sort_index()

    labels = [ETIQUETAS[i] for i in conteos.index]
    sizes  = conteos.values
    total  = int(sizes.sum())

    logger.info('Distribución categorías test:')
    for lbl, cnt in zip(labels, sizes):
        logger.info(f'  {lbl:15s} {cnt:>8,}  ({cnt/total*100:.1f}%)')

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=COLORES,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.80,
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title(
        f'Distribución de categorías de chollo\n(conjunto de prueba, {total:,} registros)',
        fontsize=13,
        pad=15,
    )

    ruta = BASE / 'images' / 'distribucion_categorias_test.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f'[OK] Gráfico guardado: {ruta}')


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main():
    grafico_distribucion_chollo_test()


if __name__ == '__main__':
    main()
