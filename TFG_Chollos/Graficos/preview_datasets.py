"""
preview_datasets.py
===================
Genera imágenes de previsualización (primeras filas + estadísticas) para los
datasets RAW (CSV de scraping) y PROCESSED (db_final_analisis.parquet).

Uso:
    python Graficos/preview_datasets.py
"""

# =============================================================================
# IMPORTS
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG

# =============================================================================
# CONSTANTES
# =============================================================================
BASE   = conseguir_ruta_general_TFG()
logger = configurar_logger(__name__)

# Columnas a mostrar según el tipo de dataset
COLS_RAW = [
    'lugar', 'titulo', 'tipo', 'estrellas',
    'valoracion_clientes', 'n_valoraciones', 'servicios',
]
COLS_PROCESSED = [
    'provincia', 'localidad', 'tipo', 'estrellas',
    'distancia_centro_km', 'tamaño_habitacion', 'precio',
]
N_FILAS = 5

# Colores
COLOR_HEADER_TEXTO = '#1a5276'
COLOR_TABLA_HEAD   = '#2c3e50'
COLOR_FILA_PAR     = '#eaf2ff'
COLOR_FILA_IMPAR   = '#ffffff'
COLOR_BORDE        = '#b0bec5'

# =============================================================================
# FUNCIONES
# =============================================================================

def _formatear_celda(valor, dtype) -> str:
    if pd.isna(valor):
        return ''
    if hasattr(dtype, 'tz') or str(dtype).startswith('datetime'):
        return pd.Timestamp(valor).strftime('%Y-%m-%d')
    if str(dtype) in ('float32', 'float64'):
        return f'{valor:.2f}'
    val_str = str(valor)
    return val_str[:30] + '…' if len(val_str) > 30 else val_str


def _contar_filas_csv(ruta: Path) -> int:
    with open(ruta, 'rb') as f:
        return sum(1 for _ in f) - 1  # descuenta la cabecera


def generar_preview(ruta: Path, ruta_salida: Path,
                    cols_mostrar: list | None = None) -> None:

    ruta = Path(ruta)

    if ruta.suffix == '.csv':
        df = pd.read_csv(ruta, sep='|', nrows=N_FILAS, on_bad_lines='skip', engine='python')
        n_registros = _contar_filas_csv(ruta)
    else:
        df = pd.read_parquet(ruta)
        n_registros = len(df)

    n_cols  = len(df.columns)
    peso_mb = ruta.stat().st_size / (1024 ** 2)

    # Seleccionar columnas: usar cols_mostrar si se pasa, sino las que coincidan del default
    candidatas = cols_mostrar if cols_mostrar is not None else (
        COLS_RAW if ruta.suffix == '.csv' else COLS_PROCESSED
    )
    cols = [c for c in candidatas if c in df.columns]
    if not cols:  # fallback: primeras N columnas
        cols = df.columns[:17].tolist()
    n_mostrar = len(cols)
    df_head   = df[cols].head(N_FILAS)

    # Formatear valores para visualización
    df_str = pd.DataFrame({
        col: [_formatear_celda(v, df[col].dtype) for v in df_head[col]]
        for col in cols
    })

    # Encabezados abreviados si son muy largos
    col_labels = [c[:14] + '…' if len(c) > 14 else c for c in cols]

    # Tamaño de figura
    fig_w = max(18, n_mostrar * 2.8)
    fig_h = 4.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')
    ax.axis('off')

    table = ax.table(
        cellText=df_str.values,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(n_mostrar)))

    for (fila, col), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor(COLOR_BORDE)
        if fila == 0:
            cell.set_facecolor(COLOR_TABLA_HEAD)
            cell.set_text_props(color='white', fontweight='bold')
        elif fila % 2 == 0:
            cell.set_facecolor(COLOR_FILA_PAR)
        else:
            cell.set_facecolor(COLOR_FILA_IMPAR)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f'[OK] Imagen guardada: {ruta_salida}')


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    generar_preview(
        ruta=BASE / 'data' / 'raw' / 'fichas' / 'resultados_booking_Málaga.csv',
        ruta_salida=BASE / 'images' / 'tabla_raw.png',
    )
    generar_preview(
        ruta=BASE / 'data' / 'processed' / 'analisis' / 'db_final_analisis.parquet',
        ruta_salida=BASE / 'images' / 'tabla_processed.png',
    )


if __name__ == '__main__':
    main()
