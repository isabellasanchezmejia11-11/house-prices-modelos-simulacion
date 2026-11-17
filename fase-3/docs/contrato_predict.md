# Contrato y diccionario de POST /predict

Contrato de las versiones `entrega-v1` y `entrega-v2`. La API recibe un objeto JSON por solicitud y responde `{"prediccion": 113650.17525962926, "version": "entrega-v1"}` para la entrada conocida. El precio se expresa en USD del conjunto histórico de Ames; no se convierte moneda ni se actualiza por inflación.

## Reglas comunes

- Los 77 campos de la tabla son obligatorios como claves. Una clave omitida produce HTTP 422; una clave con `null` está presente y se procesa según la tabla.
- `número` significa número JSON finito, sin booleanos ni cadenas numéricas. `texto` significa cadena JSON. La validación actual comprueba tipos, no todos los rangos o coherencias de dominio; el cliente debe respetar las unidades y códigos del conjunto original.
- Las unidades son las del conjunto original: pies y pies cuadrados, sin conversión automática a metros. `MSSubClass` es un código, aunque el modelo lo trata como numérico.
- Las columnas numéricas desconocidas se imputan con la mediana aprendida; las categóricas desconocidas, con la categoría más frecuente, salvo las ausencias estructurales indicadas abajo.
- Una categoría de texto nueva es aceptada: `OneHotEncoder(handle_unknown="ignore")` deja a cero el bloque correspondiente. Esto evita un error técnico, pero no garantiza una estimación fiable. No usar una categoría inventada para representar un dato desconocido.
- No enviar `SalePrice` ni claves desconocidas: HTTP 422. `Id`, `SaleType` y `SaleCondition` son opcionales y no participan en la predicción.
- La respuesta identifica el modelo mediante `version`. Solo se recibe una vivienda por solicitud y no se devuelve `Id`. La predicción por CSV de fase 2 admite varias filas y conserva su orden; esta API no admite lotes.

## Ausencia estructural y dato desconocido

En las columnas marcadas **A**, `null`, `"NA"` y `"None"` representan ausencia estructural y se convierten a la categoría `"None"`. El cliente debe usarlas solo cuando conoce que la característica no existe. En el conjunto original, por ejemplo, un sótano con exposición `"No"` es distinto de no tener sótano.

**Limitación de las versiones entregadas:** no hay un marcador independiente para “se desconoce” en las columnas A. En particular, `GarageType: null` genera `HasGarage=0` y fuerza `GarageYrBlt=0`; no significa “no sabemos si tiene garaje”. Si ese dato se desconoce, debe completarse antes de solicitar la predicción. No sustituirlo por una categoría inventada: cualquier texto distinto de `"NA"`/`"None"` hace que `HasGarage=1`. Esta documentación conserva el comportamiento del modelo entregado; permitir ambos significados exigiría diseñar y validar una nueva versión.

En `MasVnrType`, `"None"` indica ausencia de revestimiento y `null` es un dato desconocido que se imputa. `MasVnrArea=0` expresa área nula conocida y `null` un área desconocida. No convertir automáticamente esos nulos en ceros o en `"None"`.

**Leyenda de null:** A = ausencia estructural; N = desconocido, mediana aprendida; C = desconocido, categoría más frecuente; G = 0 si no hay garaje, mediana si hay garaje y se desconoce el año.

## Campos requeridos

| Campo | Significado | Tipo JSON | Unidad o escala | null |
|---|---|---|---|---|
| `MSSubClass` | Clase de vivienda | número | código numérico | N |
| `MSZoning` | Zonificación | texto | categoría | C |
| `LotFrontage` | Frente del lote a la calle | número | pies | N |
| `LotArea` | Área del lote | número | pies² | N |
| `Street` | Tipo de acceso vial | texto | categoría | C |
| `Alley` | Tipo de callejón | texto | categoría | A |
| `LotShape` | Forma del lote | texto | categoría | C |
| `LandContour` | Relieve del lote | texto | categoría | C |
| `Utilities` | Servicios disponibles | texto | categoría | C |
| `LotConfig` | Configuración del lote | texto | categoría | C |
| `LandSlope` | Pendiente del terreno | texto | categoría | C |
| `Neighborhood` | Barrio de Ames | texto | categoría | C |
| `Condition1` | Proximidad a vías u otras condiciones | texto | categoría | C |
| `Condition2` | Segunda condición de proximidad | texto | categoría | C |
| `BldgType` | Tipo de edificio | texto | categoría | C |
| `HouseStyle` | Estilo de vivienda | texto | categoría | C |
| `OverallQual` | Calidad general de materiales y acabados | número | escala 1–10 | N |
| `OverallCond` | Estado general | número | escala 1–10 | N |
| `YearBuilt` | Año de construcción | número | año | N |
| `YearRemodAdd` | Año de remodelación o ampliación | número | año | N |
| `RoofStyle` | Estilo del techo | texto | categoría | C |
| `RoofMatl` | Material del techo | texto | categoría | C |
| `Exterior1st` | Revestimiento exterior principal | texto | categoría | C |
| `Exterior2nd` | Revestimiento exterior secundario | texto | categoría | C |
| `MasVnrType` | Tipo de mampostería de revestimiento | texto | categoría | C |
| `MasVnrArea` | Área de mampostería de revestimiento | número | pies² | N |
| `ExterQual` | Calidad del material exterior | texto | categoría ordinal | C |
| `ExterCond` | Estado del material exterior | texto | categoría ordinal | C |
| `Foundation` | Tipo de cimentación | texto | categoría | C |
| `BsmtQual` | Calidad asociada a la altura del sótano | texto | categoría ordinal | A |
| `BsmtCond` | Estado del sótano | texto | categoría ordinal | A |
| `BsmtExposure` | Exposición del sótano al exterior | texto | categoría | A |
| `BsmtFinType1` | Tipo del primer acabado del sótano | texto | categoría | A |
| `BsmtFinSF1` | Área del primer acabado del sótano | número | pies² | N |
| `BsmtFinType2` | Tipo del segundo acabado del sótano | texto | categoría | A |
| `BsmtFinSF2` | Área del segundo acabado del sótano | número | pies² | N |
| `BsmtUnfSF` | Área sin terminar del sótano | número | pies² | N |
| `TotalBsmtSF` | Área total del sótano | número | pies² | N |
| `Heating` | Tipo de calefacción | texto | categoría | C |
| `HeatingQC` | Calidad y estado de calefacción | texto | categoría ordinal | C |
| `CentralAir` | Aire acondicionado central | texto | categoría Y/N | C |
| `Electrical` | Sistema eléctrico | texto | categoría | C |
| `1stFlrSF` | Área del primer piso | número | pies² | N |
| `2ndFlrSF` | Área del segundo piso | número | pies² | N |
| `LowQualFinSF` | Área terminada de baja calidad | número | pies² | N |
| `GrLivArea` | Área habitable sobre el nivel del suelo | número | pies² | N |
| `BsmtFullBath` | Baños completos en sótano | número | cantidad | N |
| `BsmtHalfBath` | Medios baños en sótano | número | cantidad | N |
| `FullBath` | Baños completos sobre el nivel del suelo | número | cantidad | N |
| `HalfBath` | Medios baños sobre el nivel del suelo | número | cantidad | N |
| `BedroomAbvGr` | Dormitorios sobre el nivel del suelo | número | cantidad | N |
| `KitchenAbvGr` | Cocinas sobre el nivel del suelo | número | cantidad | N |
| `KitchenQual` | Calidad de cocina | texto | categoría ordinal | C |
| `TotRmsAbvGrd` | Habitaciones sobre el nivel del suelo, sin baños | número | cantidad | N |
| `Functional` | Funcionalidad de la vivienda | texto | categoría | C |
| `Fireplaces` | Chimeneas | número | cantidad | N |
| `FireplaceQu` | Calidad de chimenea | texto | categoría ordinal | A |
| `GarageType` | Ubicación o tipo de garaje | texto | categoría | A |
| `GarageYrBlt` | Año de construcción del garaje | número | año | G |
| `GarageFinish` | Acabado interior del garaje | texto | categoría | A |
| `GarageCars` | Capacidad del garaje | número | vehículos | N |
| `GarageArea` | Área del garaje | número | pies² | N |
| `GarageQual` | Calidad del garaje | texto | categoría ordinal | A |
| `GarageCond` | Estado del garaje | texto | categoría ordinal | A |
| `PavedDrive` | Pavimentación de entrada | texto | categoría | C |
| `WoodDeckSF` | Área de terraza de madera | número | pies² | N |
| `OpenPorchSF` | Área de porche abierto | número | pies² | N |
| `EnclosedPorch` | Área de porche cerrado | número | pies² | N |
| `3SsnPorch` | Área de porche para tres estaciones | número | pies² | N |
| `ScreenPorch` | Área de porche con malla | número | pies² | N |
| `PoolArea` | Área de piscina | número | pies² | N |
| `PoolQC` | Calidad de piscina | texto | categoría ordinal | A |
| `Fence` | Calidad de cerca | texto | categoría | A |
| `MiscFeature` | Característica adicional | texto | categoría | A |
| `MiscVal` | Valor de característica adicional | número | USD | N |
| `MoSold` | Mes previsto de venta | número | mes 1–12 | N |
| `YrSold` | Año previsto de venta | número | año | N |

## Campos opcionales y salida

| Campo | Tipo JSON | Significado / unidad | null |
|---|---|---|---|
| `Id` | número finito | Identificador; no se devuelve en esta API | Permitido; ignorado por el modelo |
| `SaleType` | texto | Tipo de venta | Permitido; ignorado por el modelo |
| `SaleCondition` | texto | Condición de venta | Permitido; ignorado al predecir |
| `prediccion` (salida) | número | Precio estimado en USD | Nunca |
| `version` (salida) | texto | Identificador del modelo usado | Nunca |

Los códigos categóricos deben conservar la escritura del conjunto [House Prices de Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). El contrato se implementa en [`api.py`](../api.py), [`preprocessing.py`](../../fase-2/house_prices/preprocessing.py) y [`pipeline.py`](../../fase-2/house_prices/pipeline.py); el esquema por versión se conserva en sus metadatos.
