# SPI Agro — Predicción de Sequía e Inundación
### Trabajo Integrador — Ciencia de Datos

---

## Estructura del proyecto

```
spi-app/
├── api/                  ← API FastAPI (se despliega en Render)
│   ├── main.py           ← Código principal de la API
│   ├── requirements.txt  ← Dependencias Python
│   ├── Procfile          ← Comando de inicio para Render
│   └── modelo_spi_lstm.keras  ← Subir este archivo desde Colab
│
└── web/                  ← Web app (se despliega en GitHub Pages)
    ├── index.html        ← App completa (una sola página)
    └── manifest.json     ← Configuración PWA
```

---

## Paso 1 — Descargar el modelo desde Colab

Al final de la Celda 5 del notebook, ejecutá:

```python
from google.colab import files
files.download('modelo_spi_lstm.keras')
```

Guardá el archivo descargado — lo vas a necesitar para la API.

---

## Paso 2 — Subir el código a GitHub

1. Creá un repositorio nuevo en github.com (nombre sugerido: `spi-agro`)
2. Subí **toda la carpeta** del proyecto
3. El archivo `modelo_spi_lstm.keras` va dentro de la carpeta `/api`

```bash
# Si usás Git por línea de comandos:
git init
git add .
git commit -m "SPI Agro - trabajo integrador"
git remote add origin https://github.com/TU_USUARIO/spi-agro.git
git push -u origin main
```

---

## Paso 3 — Desplegar la API en Render (gratis)

1. Entrá a **render.com** y creá una cuenta gratuita
2. Click en **"New Web Service"**
3. Conectá tu cuenta de GitHub y seleccioná el repositorio `spi-agro`
4. Configurá así:
   - **Name:** `spi-agro-api`
   - **Root Directory:** `api`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click en **"Create Web Service"**
6. Esperá 3-5 minutos — Render te va a dar una URL tipo:
   `https://spi-agro-api.onrender.com`

✅ Para verificar: entrá a `https://spi-agro-api.onrender.com/estado`

---

## Paso 4 — Configurar la URL de la API en la web app

Abrí `web/index.html` y buscá esta línea (cerca del final):

```javascript
const API_URL = "https://TU-API.onrender.com";
```

Reemplazala con tu URL real de Render:

```javascript
const API_URL = "https://spi-agro-api.onrender.com";
```

Guardá el archivo y hacé commit + push a GitHub.

---

## Paso 5 — Publicar la web app en GitHub Pages (gratis)

1. En tu repositorio de GitHub, entrá a **Settings → Pages**
2. En "Source" seleccioná **"Deploy from a branch"**
3. Branch: `main`, Folder: `/web`
4. Click en **Save**
5. En 1-2 minutos tu app va a estar disponible en:
   `https://TU_USUARIO.github.io/spi-agro`

---

## Paso 6 — Generar el QR

Ejecutá esta celda en Colab (o en Python local):

```python
!pip install qrcode pillow -q
import qrcode

URL_APP = "https://TU_USUARIO.github.io/spi-agro"

qr = qrcode.QRCode(version=1, box_size=10, border=4)
qr.add_data(URL_APP)
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
img.save("qr_spi_agro.png")

from google.colab import files
files.download("qr_spi_agro.png")

print(f"QR generado para: {URL_APP}")
```

Imprimí o proyectá el QR — cualquier persona que lo escanée verá la app.

---

## Notas importantes

### Primera carga de la API (Render plan gratuito)
El plan gratuito de Render "duerme" la API después de 15 minutos sin uso.
La primera request tarda ~30 segundos en despertar el servidor.
Las siguientes son inmediatas.

**Solución para la presentación:** Entrá a la API 5 minutos antes de la clase
para que esté despierta cuando los alumnos escaneen el QR.
URL para despertar: `https://TU-API.onrender.com/estado`

### Sin modelo LSTM
Si el modelo no terminó de entrenar, la API usa predicción estadística automáticamente.
El endpoint `/estado` te dice qué modo está usando.

### Funcionamiento offline
La web app funciona como PWA — los alumnos pueden instalarla en su celular
(botón "Agregar a pantalla de inicio") y usarla sin conexión con los últimos datos.

---

## URLs finales del proyecto

| Recurso | URL |
|---|---|
| Web App | `https://TU_USUARIO.github.io/spi-agro` |
| API | `https://spi-agro-api.onrender.com` |
| Docs API | `https://spi-agro-api.onrender.com/docs` |
| Estado API | `https://spi-agro-api.onrender.com/estado` |
| Predicción SPI Santa Fe | `https://spi-agro-api.onrender.com/spi/santafe` |
