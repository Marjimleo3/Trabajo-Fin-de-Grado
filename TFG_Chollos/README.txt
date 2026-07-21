Primero, instalamos la carpeta .venv se encuentra fuera de OneDrive en 'D:\Entorno_uv_TFG_VStudio\.venv' (ya que da problemas con los hardlinks de OneDrive), abrimos una terminal en esa carpeta y ejecutamos lo siguiente:
uv venv --python 3.12     ----> '--python 3.12' es opcional

Para asociar las dependencias del entorno uv de la carpeta en la que estamos, a la carpeta .venv fuera de OneDrive (ya que genera problemas) y se active también automáticamente, hacemos esto:
notepad $PROFILE  --> Abrimos un bloc de notas
$env:UV_PROJECT_ENVIRONMENT = "D:\Entorno_uv_TFG_VStudio\.venv"
D:\Entorno_uv_TFG_VStudio\.venv\Scripts\Activate.ps1  --> Guardamos estas 2 líneas (Si hay algo distinto es porque ahí se guardan todas las vinculaciones y activaciones automáticas)
uv sync  --> Ejecutamos esto en la terminal

Abrir CMD en la carpeta de nuestro trabajo y activar el entorno fuera de OneDrive:
D:\Entorno_uv_TFG_VStudio\.venv\Scripts\activate

Una vez vinculadas las carpetas y activado el entorno, para instalar nuevos paquetes en uv, debemos proceder de la siguiente forma para que no se creen nuevos .venv en la carpeta de nuestro TFG dentro de OneDrive:
uv add streamlit pandas --active --link-mode=copy



