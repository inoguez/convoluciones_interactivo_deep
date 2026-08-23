import unittest

import numpy as np

from experimentos_kernels import aplicar_kernel, crear_imagen_prueba, kernels_evaluados


class TestKernels(unittest.TestCase):
    def test_imagen_reproducible(self):
        primera = crear_imagen_prueba(256)
        segunda = crear_imagen_prueba(256)

        self.assertEqual(primera.shape, (256, 256))
        self.assertEqual(primera.dtype, np.uint8)
        np.testing.assert_array_equal(primera, segunda)

    def test_exactamente_cinco_kernels(self):
        kernels = kernels_evaluados()

        self.assertEqual(len(kernels), 5)
        self.assertEqual(
            sum(nombre.startswith("Personalizado") for nombre in kernels),
            2,
        )

    def test_convolucion_conserva_forma(self):
        imagen = crear_imagen_prueba(64)

        resultado = aplicar_kernel(imagen, np.eye(3))

        self.assertEqual(resultado.shape, imagen.shape)
        self.assertEqual(resultado.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
