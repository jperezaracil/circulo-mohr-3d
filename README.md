# Círculos de Mohr 3D — Aplicación interactiva

Aplicación didáctica para visualizar el estado tensional en un punto de un sólido y su representación sobre el círculo de Mohr (caso 3D general; el caso 2D se obtiene como caso particular).

## Qué muestra

- **Cubo 3D girado** con las flechas de tensión normal (rojo) y cortante (azul) sobre cada cara.
- **Diagrama de Mohr 3D** con los tres semicírculos, la región admisible sombreada y los puntos `(σ_n, |τ|)` de las tres caras del cubo girado.
- Tensiones principales σ₁ ≥ σ₂ ≥ σ₃ y cortante máxima τ_máx.

## Controles

- 6 sliders para el tensor inicial: σx, σy, σz, τxy, τxz, τyz.
- 3 sliders para los ángulos de rotación: α (alrededor de x), β (y), γ (z).
- Un selector de presets: tracción uniaxial, cortante puro, hidrostático, cortante 3D genérico.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Versión Jupyter

El archivo `circulo_de_mohr.ipynb` contiene la misma visualización en formato notebook, con explicaciones más detalladas de las ecuaciones.
