#CADA VEZ QUE SE RETOCA UNA FUNCIÓN, HAY QUE REINICIAR EL KERNEL

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from dotenv import load_dotenv
from pathlib import Path
import os
import logging

def conseguir_ruta_general_TFG():
    '''
    Cargamos las constantes del archivo .env como variables de entorno del sistema y extraemos todas las variables de entorno.
    *Nota. 'Path' se usa para poder concatenar más cómodamente la ruta (convierte un string en un objeto de ruta inteligente) y 'getenv' significa get environment variable
    '''
    load_dotenv()  #Carga las rutas del archivo .env como variables de entorno del sistema
    base_env = os.getenv("BASE")    
    if base_env:
        return Path(base_env)
    return Path(__file__).parent  # Fallback para Streamlit Cloud: ruta relativa al propio utils.py



def configurar_logger(nombre: str = __name__) -> logging.Logger:
    '''
    Configura la salida del logger
    '''
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    logger = logging.getLogger(nombre)
    return logger



def get_outliers(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    big_inf = Q1 - 1.5 * IQR
    big_sup = Q3 + 1.5 * IQR
    outliers = df[(df[columna] < big_inf) | (df[columna] > big_sup)]
    return outliers, big_inf, big_sup



def denegar_cookies(driver, timeout=2):
    '''
    Deniega las cookies. Pulsa el botón 'Rechazar'
    '''
    try:
        # Esperar a que el banner sea visible (no solo presente en el DOM)
        WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.ID, "onetrust-banner-sdk"))
        )

        # Esperar a que el botón sea clickable (no solo presente)
        boton = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.ID, "onetrust-reject-all-handler"))
        )

        driver.execute_script("arguments[0].click();", boton)

        WebDriverWait(driver, timeout).until(
            EC.invisibility_of_element_located((By.ID, "onetrust-banner-sdk"))
        )

        print("[OK] Cookies cerradas")

    except TimeoutException:
        print("ℹ️ Banner de cookies no apareció o ya estaba cerrado")

