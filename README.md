# Experimento de kernels de convolución

Este fork conserva la aplicación Streamlit original y añade un experimento reproducible para comparar dos kernels propuestos y tres filtros predeterminados. La imagen de entrada se genera con código y contiene gradientes, figuras, texto y una diagonal, por lo que no depende de archivos externos.

## Ejecución

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest -v test_experimentos.py
python3 experimentos_kernels.py
```

La comparación se guarda en `resultados/kernels_comparacion.png` y las estadísticas en `resultados/metricas_kernels.json`.

## Kernels propuestos

### 1. Detector diagonal

```text
 2   1   0
 1   0  -1
 0  -1  -2
```

La suma es cero, así que las zonas uniformes producen respuestas pequeñas. Los pesos positivos y negativos están enfrentados en diagonal: el filtro responde con fuerza cuando existe una transición de intensidad orientada del noroeste al sureste. En la imagen se destacan la línea diagonal y ciertas esquinas de las figuras.

### 2. Realce de paso alto

```text
-1  -1  -1
-1   9  -1
-1  -1  -1
```

Conserva el píxel central y resta sus ocho vecinos. Esto aumenta el contraste local y hace más nítidos bordes y detalles pequeños, pero también amplifica ruido o cambios bruscos no deseados.

## Tres kernels predeterminados

- **Gaussiano:** calcula un promedio ponderado que da más importancia al centro. Reduce variaciones locales y ruido, a cambio de suavizar detalles.
- **Sobel X:** compara la intensidad entre izquierda y derecha. Detecta cambios en el eje horizontal y, por eso, resalta principalmente bordes verticales.
- **Enfoque:** suma cinco veces el centro y resta sus vecinos directos. Refuerza contornos conservando el brillo medio porque sus coeficientes suman uno.

## Lo que entendí de la explicación de la IA

Un kernel no “reconoce” objetos por sí mismo: mide un patrón local en cada vecindario. Una suma cercana a uno tiende a conservar el brillo; una suma cero elimina regiones constantes y deja visibles las transiciones. La posición de los valores positivos y negativos determina la orientación a la que responde el filtro. Los kernels de una CNN parten de esta misma operación, pero sus coeficientes se aprenden durante el entrenamiento en lugar de establecerse manualmente.

## Alcance

La prueba usa una sola imagen sintética para que todos puedan reproducir exactamente el resultado. Las conclusiones describen el comportamiento del filtro; no evalúan calidad sobre un conjunto amplio de imágenes.
