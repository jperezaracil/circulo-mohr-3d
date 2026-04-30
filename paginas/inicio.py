"""Página de inicio."""

import streamlit as st

st.title("Resistencia de Materiales — Aplicaciones interactivas")

st.markdown(
    """
    Bienvenido. Esta aplicación reúne herramientas didácticas interactivas
    para Resistencia de Materiales. Selecciona la herramienta en la
    **barra lateral izquierda**.

    ---

    ### Círculos de Mohr 3D

    Visualiza el estado tensional general en un punto de un sólido y su
    representación sobre los tres círculos de Mohr. Con sliders para las
    seis componentes del tensor y los tres ángulos de rotación. El caso
    2D (estado plano) se obtiene como caso particular.

    Conceptos clave: tensor de tensiones, transformación de tensiones,
    tensiones principales, círculos de Mohr 3D, región admisible,
    cortante máxima.

    ### Pandeo de columnas

    Calcula la carga crítica de una columna y compara las cuatro
    fórmulas clásicas (Euler, Johnson, Tetmajer, Rankine-Gordon)
    sobre el diagrama σ-λ. Incluye un catálogo de perfiles laminados
    (IPE, HEB, HEA, UPN), secciones paramétricas y los cuatro casos
    clásicos de condiciones de contorno.

    Conceptos clave: esbeltez, radio de giro, longitud de pandeo,
    coeficiente K, esbeltez límite, régimen elástico vs plástico,
    coeficiente de seguridad.

    ---

    *Material didáctico abierto. Código fuente en
    [GitHub](https://github.com/jperezaracil/circulo-mohr-3d).*
    """
)
