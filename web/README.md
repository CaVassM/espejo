# Web de Aula Evidente

Dos páginas estáticas, sin compilación ni dependencias que instalar:

```
web/
├── index.html        landing
├── assets/           estilos, scripts y logotipos (compartidos)
├── app/
│   ├── index.html    la ficha de evidencias
│   ├── charts.js     las gráficas (SVG a mano, sin librerías)
│   ├── minedu.js     el mapeo de las cinco rúbricas
│   └── demo/         los datos que se publican
└── vercel.json       configuración del despliegue
```

## Verla en local

Desde la raíz del repositorio:

```bash
python services/serve_demo.py
```

Abre `http://localhost:8000/`. En local la ficha lee los resultados recién
producidos por el motor (`outputs/clase/`) y reproduce el video de `samples/`.

## Publicarla en Vercel

En Vercel: **Add New → Project**, elegir este repositorio y configurar:

| Ajuste | Valor |
|---|---|
| Root Directory | `web` |
| Framework Preset | Other |
| Build Command | activar *Override* y dejarlo vacío |
| Output Directory | dejarlo por defecto |

No hace falta nada más: `web/vercel.json` trae las cabeceras y el
`trailingSlash`, y no hay paso de compilación que pueda fallar.

### Por qué el Root Directory importa

Vercel no deja que la aplicación lea archivos fuera del Root Directory, ni usar
`..` para subir de nivel. Por eso la ficha publicada **no** lee `outputs/` ni
`samples/`, sino los datos de `app/demo/`. La página detecta dónde está por su
propia ruta y elige el origen correspondiente, así que el mismo archivo sirve en
local y publicado sin tocar nada.

## Qué se publica y qué no

| | |
|---|---|
| Se publica | el reporte (`app/demo/class_report.json`) y el mapa de calor |
| **No** se publica | **el video de la clase** |

El video queda fuera por dos razones, y las dos son definitivas: la grabación
tiene **estudiantes menores de edad** y el repositorio es público, y además pesa
121 MB, por encima del límite de 100 MB de GitHub. `*.mp4` está en `.gitignore`.

Sin video, la ficha publicada retira el reproductor y lo explica en pantalla.
Todo lo demás sigue: cada evidencia conserva su minuto y su cita textual, las
cinco rúbricas, las gráficas, las recomendaciones y la descarga en PDF.

El mapa de calor sí se publica: va sobre un fondo desenfocado hasta el punto de
que no se reconoce a nadie, y el reporte reemplaza los nombres por `[NOMBRE]`.

## Para demostrar el flujo completo con video

Hay que usarlo en local, donde el video no sale de la máquina. Si en algún
momento se quiere publicar una demostración con video, tiene que ser una clase
grabada **con personas adultas que hayan dado su consentimiento**, como pide el
plan maestro del equipo.
