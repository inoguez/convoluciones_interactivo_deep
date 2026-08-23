"""Experimento reproducible para comparar kernels de convolución."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import convolve


SALIDA = Path(__file__).resolve().parent / "resultados"


def crear_imagen_prueba(tamano: int = 256) -> np.ndarray:
    """Crea una imagen determinista con gradientes, curvas, texto y bordes."""
    if tamano < 32:
        raise ValueError("El tamaño debe ser al menos 32")

    gradiente = np.tile(np.linspace(20, 210, tamano, dtype=np.uint8), (tamano, 1))
    lienzo = Image.fromarray(gradiente, mode="L")
    dibujo = ImageDraw.Draw(lienzo)
    margen = tamano // 10
    dibujo.rectangle(
        (margen, margen, tamano // 2, tamano // 2),
        outline=245,
        width=max(2, tamano // 64),
        fill=75,
    )
    dibujo.ellipse(
        (tamano // 2, margen, tamano - margen, tamano // 2),
        outline=15,
        width=max(2, tamano // 64),
        fill=190,
    )
    dibujo.line(
        (margen, tamano - margen, tamano - margen, tamano // 2),
        fill=250,
        width=max(3, tamano // 48),
    )
    dibujo.text(
        (margen, tamano * 3 // 5),
        "CNN",
        fill=30,
        font=ImageFont.load_default(),
        stroke_width=1,
        stroke_fill=235,
    )
    return np.asarray(lienzo, dtype=np.uint8)


def kernels_evaluados() -> dict[str, np.ndarray]:
    """Devuelve dos kernels propios y tres kernels predeterminados."""
    return {
        "Personalizado - diagonal": np.array(
            [[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=float
        ),
        "Personalizado - paso alto": np.array(
            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=float
        ),
        "Gaussiano": np.array(
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float
        )
        / 16,
        "Sobel X": np.array(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float
        ),
        "Enfoque": np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float
        ),
    }


def aplicar_kernel(imagen: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Aplica una convolución 2D y produce una imagen visible de 8 bits."""
    if imagen.ndim != 2:
        raise ValueError("La imagen debe estar en escala de grises")
    if kernel.ndim != 2 or min(kernel.shape) < 1:
        raise ValueError("El kernel debe ser una matriz 2D no vacía")

    bruto = convolve(imagen.astype(float), kernel.astype(float), mode="reflect")
    if np.isclose(kernel.sum(), 0):
        bruto = np.abs(bruto)
    return np.clip(bruto, 0, 255).astype(np.uint8)


def generar_lamina(directorio: Path = SALIDA) -> Path:
    """Ejecuta los cinco kernels y guarda una comparación y sus estadísticas."""
    directorio.mkdir(parents=True, exist_ok=True)
    imagen = crear_imagen_prueba()
    resultados = {
        nombre: aplicar_kernel(imagen, kernel)
        for nombre, kernel in kernels_evaluados().items()
    }

    figura, ejes = plt.subplots(2, 3, figsize=(12, 8))
    paneles = [("Imagen original", imagen), *resultados.items()]
    for eje, (titulo, panel) in zip(ejes.flat, paneles):
        eje.imshow(panel, cmap="gray", vmin=0, vmax=255)
        eje.set_title(titulo, fontsize=11)
        eje.axis("off")
    figura.suptitle("Comparación reproducible de cinco kernels", fontsize=16)
    figura.tight_layout(rect=(0, 0, 1, 0.96))
    ruta = directorio / "kernels_comparacion.png"
    figura.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(figura)

    estadisticas = {
        nombre: {
            "media": round(float(panel.mean()), 3),
            "desviacion": round(float(panel.std()), 3),
            "minimo": int(panel.min()),
            "maximo": int(panel.max()),
        }
        for nombre, panel in resultados.items()
    }
    (directorio / "metricas_kernels.json").write_text(
        json.dumps(estadisticas, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ruta


if __name__ == "__main__":
    salida = generar_lamina()
    print(f"Comparación creada: {salida}")
