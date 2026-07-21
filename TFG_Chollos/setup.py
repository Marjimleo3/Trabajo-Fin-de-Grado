# Hace que TFG_Chollos sea importable como paquete en Streamlit Cloud.
# Sin este archivo, los imports "from TFG_Chollos.X import ..." fallarían en la nube.
from setuptools import setup

setup(
    name='TFG-Chollos',
    version='0.1',
    packages=['TFG_Chollos', 'TFG_Chollos.Graficos', 'TFG_Chollos.Scraping'],
    package_dir={
        'TFG_Chollos': '.',
        'TFG_Chollos.Graficos': 'Graficos',
        'TFG_Chollos.Scraping': 'Scraping',
    },
)
