# Fase 4 · Monitoreo de la API

Esta fase incorpora el registro de solicitudes de predicción y el cálculo de métricas operativas para la API REST House Prices desarrollada en la fase 3.

## 1. Registro de solicitudes

Se implementó `fase-4/registro.py` para almacenar automáticamente las solicitudes realizadas al endpoint `POST /predict`, incluidas aquellas rechazadas por validación.

El archivo `fase-3/api.py` incorpora un middleware que registra cada solicitud con los siguientes campos:

- `fecha_hora`: fecha y hora de la solicitud en UTC.
- `estado_http`: código HTTP de respuesta.
- `version`: versión del modelo utilizada, si corresponde.
- `prediccion`: resultado generado, si corresponde.

Los registros se almacenan en `fase-4/registros/solicitudes.csv` utilizando el modo de escritura `append`, que conserva la información anterior.

## 2. Métricas de operación

El script `fase-4/monitoreo.py` lee el registro CSV y calcula:

- Total de solicitudes registradas.
- Total de solicitudes fallidas, considerando los códigos HTTP entre 400 y 599.
- Porcentaje de solicitudes fallidas respecto al total.

Para ejecutarlo desde la raíz del repositorio:

```powershell
python fase-4/monitoreo.py
```

También es posible especificar otro archivo:

```powershell
python fase-4/monitoreo.py --registro ruta/al/archivo.csv
```

Cuando no existen solicitudes, el script presenta total 0 y el mensaje `sin solicitudes`.

## 3. Pruebas automatizadas

Las pruebas se encuentran en `fase-4/tests/test_monitoreo.py`.

Desde la raíz del repositorio:

```powershell
python -m pip install -r fase-3/requirements-dev.txt
python -m pytest fase-4/tests/test_monitoreo.py -v
```

Las cuatro pruebas verifican el registro de predicciones válidas, el registro de solicitudes rechazadas, el cálculo de indicadores y el comportamiento ante un archivo vacío.

Resultado obtenido: **4 pruebas aprobadas**.

Para comprobar las fases 3 y 4 conjuntamente:

```powershell
python -m pytest fase-3/tests fase-4/tests -v
```

Resultado obtenido: **7 pruebas aprobadas**.

## 4. Ejecución mediante Docker

Construir la imagen desde la raíz del repositorio:

```powershell
docker build -f fase-3/Dockerfile -t house-prices-api-fase4 .
```

Ejecutar el contenedor en PowerShell:

```powershell
docker run --rm --name house-prices-api-fase4 -p 8000:8000 --mount "type=bind,source=$((Resolve-Path 'fase-1').Path),target=/app/fase-1,readonly" --mount "type=bind,source=$((Resolve-Path 'fase-2/models').Path),target=/app/fase-2/models" --mount "type=bind,source=$((Resolve-Path 'fase-4/registros').Path),target=/app/fase-4/registros" house-prices-api-fase4
```

El montaje de `fase-4/registros` permite conservar el historial de solicitudes aunque se detenga el contenedor.

La documentación interactiva de la API está disponible en `http://localhost:8000/docs`.

## 5. Resultados de comprobación

Durante las pruebas locales y en Docker se registraron cinco solicitudes: dos exitosas con HTTP 200 y tres rechazadas con HTTP 422.

Los indicadores obtenidos fueron:

- Total de solicitudes: 5.
- Solicitudes fallidas: 3.
- Porcentaje de fallos: 60,0 %.

Las solicitudes rechazadas se realizaron intencionalmente para comprobar la validación de entradas. Esta tasa representa el resultado de las pruebas y no mide la calidad predictiva del modelo.

## 6. Estado de implementación

Se completaron y comprobaron los puntos 1 y 2 de la fase 4: registro de solicitudes y cálculo de métricas operativas.

Los puntos 3 (conservación de versiones y política de reentrenamiento) y 4 (reporte final) quedan pendientes de integración por el equipo.
