# Fase 4 · Monitoreo de la API

La API registra cada solicitud a `/predict`, incluidas las que se rechazan por validación. `registro.py` agrega una fila por intento con la fecha y hora en UTC, el código HTTP, la versión utilizada y la predicción. Si la solicitud se rechaza antes de usar el modelo, los dos últimos campos quedan vacíos. El encabezado se escribe al crear el CSV.

## Iniciar la API y consultar el historial

Los comandos se ejecutan en PowerShell desde la raíz del repositorio. Para la ejecución local, instala primero las dependencias siguiendo el [README de fase 3](../fase-3/README.md).

```powershell
$env:MODEL_VERSION = "entrega-v2"
python -m uvicorn api:app --app-dir fase-3 --host 0.0.0.0 --port 8000
```

El historial se guarda en `fase-4/registros/solicitudes.csv` y se conserva entre ejecuciones. Si se configura `REGISTRO_PATH` antes de arrancar la API, se utiliza la ruta indicada. El script de monitoreo cuenta las filas de datos y considera fallidas las respuestas con códigos HTTP del 400 al 599.

```powershell
python fase-4/monitoreo.py
python fase-4/monitoreo.py --registro fase-4/registros/muestra_reporte.csv
```

Con un archivo vacío, el total es 0 y el porcentaje se muestra como `sin solicitudes`.

## Docker y persistencia

Construye la imagen desde la raíz del repositorio. La API escucha en el puerto 8000, y el contenedor usa montajes para los datos de entrenamiento, los modelos y el historial.

```powershell
docker build -f fase-3/Dockerfile -t house-prices-api-fase4 .
docker run --rm --name house-prices-api-fase4 -p 8000:8000 --mount "type=bind,source=$((Resolve-Path 'fase-1').Path),target=/app/fase-1,readonly" --mount "type=bind,source=$((Resolve-Path 'fase-2/models').Path),target=/app/fase-2/models" --mount "type=bind,source=$((Resolve-Path 'fase-4/registros').Path),target=/app/fase-4/registros" -e MODEL_VERSION=entrega-v2 house-prices-api-fase4
```

El montaje de `fase-4/registros` conserva el CSV al detener el contenedor. Para repetir el ensayo sobre un archivo vacío, agrega `-e REGISTRO_PATH=/app/fase-4/registros/solicitudes-ensayo.csv` antes del nombre de la imagen y usa un nombre que aún no exista en esa carpeta. En ejecución local puedes configurar `$env:REGISTRO_PATH = Join-Path (Get-Location).Path "fase-4/registros/solicitudes-ensayo.csv"` antes de iniciar Uvicorn. Los registros de un ensayo posterior se consultan con `python fase-4/monitoreo.py --registro fase-4/registros/solicitudes-ensayo.csv`.

## Reporte

Se inició la API con `entrega-v2` y un CSV vacío. Las tres solicitudes se enviaron a `POST /predict` a partir de [example_request.json](../fase-3/data/example_request.json). Se reinició la API tras la primera predicción para comprobar que las siguientes filas se agregaran al mismo archivo.

| Solicitud | Datos enviados | HTTP | Versión | Predicción |
| --- | --- | --- | --- | --- |
| 1 | JSON de ejemplo, con `GrLivArea` de 896 | 200 | `entrega-v2` | 113.650,18 |
| 2 | Mismo JSON, con `GrLivArea` de 996 | 200 | `entrega-v2` | 116.877,85 |
| 3 | JSON de la segunda solicitud sin `GrLivArea` | 422 | | |

Al ejecutar `monitoreo.py` sobre [muestra_reporte.csv](registros/muestra_reporte.csv), se obtuvieron **3 solicitudes y 33,3 % de fallos**. La única solicitud fallida fue la que omitió `GrLivArea` de forma intencional para comprobar la validación. La tasa mide los fallos de las solicitudes de este ensayo.

### Versiones y retorno a la anterior

- `entrega-v2` fue la versión activa del ensayo. Está en [fase-2/models/entrega-v2](../fase-2/models/entrega-v2/).
- `entrega-v1` es la versión anterior y la predeterminada cuando se inicia sin configurar `MODEL_VERSION`. Está en [fase-2/models/entrega-v1](../fase-2/models/entrega-v1/).

Para volver a `entrega-v1`, detén la API, ejecuta `$env:MODEL_VERSION = "entrega-v1"` y vuelve a iniciarla con el mismo comando de Uvicorn. En Docker, detén el contenedor y vuelve a crearlo con los mismos montajes y `-e MODEL_VERSION=entrega-v1`. Se verificó por separado que la API devolviera `"version": "entrega-v1"` al aplicar el cambio.

### Política de reentrenamiento

Cuando el equipo reúna 100 nuevos registros de viviendas con su precio de venta real, la persona responsable del modelo revisará el desempeño y ejecutará `POST /train` con los datos preparados para generar una versión candidata. El equipo calculará el MAE de la versión activa y de la candidata sobre las mismas viviendas reservadas para evaluación, excluidas del entrenamiento, y activará la candidata mediante `MODEL_VERSION` solo si reduce el MAE.

## Pruebas

Desde la raíz, instala las dependencias de desarrollo y ejecuta pytest.

```powershell
python -m pip install -r fase-3/requirements-dev.txt
python -m pytest -q
```

Resultado de la revisión local con Python 3.12: **20 pruebas aprobadas**, incluidas las 7 de las fases 3 y 4. Las pruebas de la API escriben en archivos temporales para que el historial de operación y la muestra del reporte se puedan consultar después de cada ejecución.
