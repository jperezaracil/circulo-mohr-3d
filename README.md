# Resistencia de Materiales — Apps interactivas

Aplicación didáctica multipágina con dos herramientas:

1. **Círculos de Mohr 3D** — visualiza el estado tensional general en
   un punto y su representación sobre los tres círculos de Mohr.
   El caso 2D se obtiene como caso particular.
2. **Pandeo de columnas** — calcula la carga crítica de pandeo y
   compara las cuatro fórmulas clásicas (Euler, Johnson, Tetmajer,
   Rankine-Gordon) sobre el diagrama σ-λ. Incluye catálogo de
   perfiles laminados (IPE, HEB, HEA, UPN), secciones paramétricas
   y los cuatro casos clásicos de condiciones de contorno.

## Estructura

```
streamlit_app.py              ← punto de entrada multipágina (st.navigation)
paginas/
  inicio.py                   ← página de bienvenida
  mohr.py                     ← círculos de Mohr (autocontenido)
  pandeo.py                   ← UI Streamlit del pandeo

pandeo_perfiles.py            ← catálogo IPE/HEB/HEA/UPN
pandeo_materiales.py          ← catálogo de materiales (acero, Al, madera)
pandeo_calculos.py            ← fórmulas (Euler, Johnson, Tetmajer, Rankine)
pandeo_graficas.py            ← funciones de matplotlib (columna + curvas σ-λ)

circulo_de_mohr.ipynb         ← notebook Jupyter (versión de aula offline)
```

Cada concepto vive en su archivo: catálogos separados de la lógica,
y la lógica separada de la UI y de las funciones de dibujo.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Despliegue

La app está pensada para [Streamlit Community Cloud](https://share.streamlit.io):
basta con apuntar a este repo, rama `main`, archivo principal
`app_streamlit.py`. Las dos herramientas se sirven detrás de la misma URL,
seleccionables desde la barra lateral.
