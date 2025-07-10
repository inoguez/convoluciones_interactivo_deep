import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import pandas as pd

def apply_convolution(image, kernel, color_mode="grayscale"):
    """
    Aplica convolución a una imagen usando el kernel especificado
    
    Args:
        image: imagen de entrada
        kernel: kernel de convolución
        color_mode: "grayscale" o "color"
    
    Returns:
        tuple: (imagen_original_procesada, imagen_convolucionada)
    """
    
    if color_mode == "grayscale":
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        # Aplicar convolución
        convolved = convolve(gray_image.astype(np.float32), kernel, mode='constant', cval=0.0)
        
        # Normalizar resultado para visualización
        convolved_normalized = np.clip(convolved, 0, 255).astype(np.uint8)
        
        return gray_image, convolved_normalized
    
    else:  # color_mode == "color"
        # Trabajar con imagen en color
        if len(image.shape) == 3:
            # Aplicar convolución a cada canal por separado
            convolved_channels = []
            for channel in range(image.shape[2]):
                channel_conv = convolve(image[:,:,channel].astype(np.float32), kernel, mode='constant', cval=0.0)
                convolved_channels.append(channel_conv)
            
            # Combinar canales
            convolved = np.stack(convolved_channels, axis=2)
        else:
            # Si la imagen es en escala de grises, aplicar convolución directamente
            convolved = convolve(image.astype(np.float32), kernel, mode='constant', cval=0.0)
        
        # Normalizar resultado para visualización
        convolved_normalized = np.clip(convolved, 0, 255).astype(np.uint8)
        
        return image, convolved_normalized

def create_kernel_grid(size):
    """
    Crea una grilla para introducir valores del kernel
    """
    kernel = np.zeros((size, size))
    
    st.write(f"Introduce los valores del kernel {size}x{size}:")
    
    # Crear columnas para la grilla
    cols = st.columns(size)
    
    for i in range(size):
        for j in range(size):
            with cols[j]:
                kernel[i, j] = st.number_input(
                    f"[{i},{j}]",
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    key=f"kernel_{i}_{j}"
                )
    
    return kernel

def get_predefined_kernels():
    """
    Retorna diccionario con kernels predefinidos
    """
    kernels = {
        "Identidad": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        "Desenfoque": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
        "Detección de bordes (Sobel X)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "Detección de bordes (Sobel Y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        "Laplaciano": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        "Enfoque": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "Relieve": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        "Desenfoque Gaussiano": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    }
    return kernels

def main():
    st.set_page_config(page_title="Aplicador de Convolución", layout="wide")
    
    st.title("Aplicador de Convolución de Imágenes")
    st.markdown("Sube una imagen y aplica diferentes kernels de convolución para ver los efectos")
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("Configuración")
        
        # Selección de modo
        mode = st.radio(
            "Modo de kernel:",
            ["Predefinidos", "Personalizado"],
            help="Elige entre kernels predefinidos o crea tu propio kernel"
        )
        
        # Selección de modo de color
        color_mode = st.radio(
            "Modo de procesamiento:",
            ["Escala de grises", "Color"],
            help="Elige si procesar la imagen en escala de grises o mantener los colores"
        )
        
        color_mode_key = "grayscale" if color_mode == "Escala de grises" else "color"
        
        if mode == "Personalizado":
            kernel_size = st.selectbox(
                "Tamaño del kernel:",
                [3, 5, 7],
                help="Tamaño del kernel (siempre cuadrado)"
            )
    
    # Área principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Subir Imagen")
        uploaded_file = st.file_uploader(
            "Selecciona una imagen",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Formatos soportados: PNG, JPG, JPEG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            # Cargar y mostrar imagen original
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            st.subheader("Imagen Original")
            st.image(image, caption="Imagen cargada", use_column_width=True)
            
            st.write(f"Dimensiones: {image_array.shape}")
    
    with col2:
        st.header("Configurar Kernel")
        
        if mode == "Predefinidos":
            kernels = get_predefined_kernels()
            selected_kernel = st.selectbox(
                "Selecciona un kernel:",
                list(kernels.keys()),
                help="Kernels predefinidos para diferentes efectos"
            )
            kernel = kernels[selected_kernel]
            
            # Mostrar kernel seleccionado
            st.subheader("Kernel seleccionado:")
            st.write(pd.DataFrame(kernel).round(3))
            
        else:
            # Modo personalizado
            kernel = create_kernel_grid(kernel_size)
            
            # Botón para normalizar kernel
            if st.button("Normalizar kernel"):
                kernel_sum = np.sum(kernel)
                if kernel_sum != 0:
                    kernel = kernel / kernel_sum
                    st.success("Kernel normalizado!")
                else:
                    st.warning("No se puede normalizar: la suma es 0")
        
        # Mostrar información del kernel
        st.subheader("Información del kernel:")
        st.write(f"Suma de elementos: {np.sum(kernel):.3f}")
        st.write(f"Valor máximo: {np.max(kernel):.3f}")
        st.write(f"Valor mínimo: {np.min(kernel):.3f}")
    
    # Aplicar convolución si hay imagen cargada
    if uploaded_file is not None:
        st.header("Resultado de la Convolución")
        
        if st.button("Aplicar Convolución", type="primary"):
            with st.spinner("Aplicando convolución..."):
                processed_image, convolved_image = apply_convolution(image_array, kernel, color_mode_key)
                
                # Mostrar resultados
                if color_mode_key == "grayscale":
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Original")
                        st.image(image_array, caption="Imagen original", use_column_width=True)
                    
                    with col2:
                        st.subheader("Escala de grises")
                        st.image(processed_image, caption="Imagen en escala de grises", use_column_width=True)
                    
                    with col3:
                        st.subheader("Convolución aplicada")
                        st.image(convolved_image, caption="Resultado de la convolución", use_column_width=True)
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original")
                        st.image(image_array, caption="Imagen original", use_column_width=True)
                    
                    with col2:
                        st.subheader("Convolución aplicada")
                        st.image(convolved_image, caption="Resultado de la convolución en color", use_column_width=True)
                
                # Estadísticas del resultado
                st.subheader("Estadísticas del resultado")
                col1, col2, col3, col4 = st.columns(4)
                
                if color_mode_key == "grayscale":
                    # Estadísticas para escala de grises
                    with col1:
                        st.metric("Valor promedio", f"{np.mean(convolved_image):.2f}")
                    with col2:
                        st.metric("Desviación estándar", f"{np.std(convolved_image):.2f}")
                    with col3:
                        st.metric("Valor máximo", f"{np.max(convolved_image)}")
                    with col4:
                        st.metric("Valor mínimo", f"{np.min(convolved_image)}")
                else:
                    # Estadísticas para color (promedio de todos los canales)
                    if len(convolved_image.shape) == 3:
                        mean_val = np.mean(convolved_image)
                        std_val = np.std(convolved_image)
                        max_val = np.max(convolved_image)
                        min_val = np.min(convolved_image)
                    else:
                        mean_val = np.mean(convolved_image)
                        std_val = np.std(convolved_image)
                        max_val = np.max(convolved_image)
                        min_val = np.min(convolved_image)
                    
                    with col1:
                        st.metric("Valor promedio", f"{mean_val:.2f}")
                    with col2:
                        st.metric("Desviación estándar", f"{std_val:.2f}")
                    with col3:
                        st.metric("Valor máximo", f"{max_val}")
                    with col4:
                        st.metric("Valor mínimo", f"{min_val}")
                
    # Información adicional
    with st.expander("Información sobre convolución"):
        st.markdown("""
        ### ¿Qué es la convolución?
        La convolución es una operación matemática que combina dos funciones. En procesamiento de imágenes, 
        se usa para aplicar filtros que pueden:
        
        - **Desenfocar** la imagen (filtros de paso bajo)
        - **Detectar bordes** (filtros de paso alto)
        - **Enfocar** o mejorar detalles
        - **Crear efectos especiales**
        
        ### Modos de procesamiento:
        - **Escala de grises**: Convierte la imagen a escala de grises antes de aplicar la convolución
        - **Color**: Aplica la convolución a cada canal de color por separado (RGB)
        
        ### Kernels comunes:
        - **Identidad**: No modifica la imagen
        - **Desenfoque**: Suaviza la imagen promediando píxeles vecinos
        - **Sobel**: Detecta bordes en direcciones específicas
        - **Laplaciano**: Detecta bordes en todas las direcciones
        - **Enfoque**: Aumenta el contraste en los bordes
        
        ### Consejos:
        - Kernels con suma = 1 mantienen el brillo promedio
        - Kernels con suma = 0 detectan cambios (bordes)
        - Valores negativos pueden crear efectos de inversión
        - El modo color preserva la información cromática de la imagen
        """)

if __name__ == "__main__":
    main()