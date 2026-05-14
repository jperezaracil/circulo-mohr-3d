"""
Resistencia de Materiales — Apps interactivas
=============================================
Punto de entrada multipágina (Streamlit). Organiza varias herramientas
didácticas detrás de una sola URL.

Lanzar localmente:
    streamlit run app_streamlit.py
"""

import streamlit as st

st.set_page_config(
    page_title="Resistencia de Materiales — Apps",
    layout="wide",
    page_icon="◯",
)

inicio   = st.Page("paginas/inicio.py",
                   title="Inicio", icon=":material/home:", default=True)
mohr     = st.Page("paginas/mohr.py",
                   title="Círculos de Mohr 3D", icon=":material/circle:")
mohr_2d  = st.Page("paginas/mohr_2d.py",
                   title="Mohr 2D — Sección YZ", icon=":material/blur_circular:")
ejercicios = st.Page("paginas/ejercicios.py",
                     title="Ejercicios de Mohr",
                     icon=":material/school:")
pandeo   = st.Page("paginas/pandeo.py",
                   title="Pandeo de columnas", icon=":material/architecture:")

pg = st.navigation([inicio, mohr, mohr_2d, ejercicios, pandeo])
pg.run()
