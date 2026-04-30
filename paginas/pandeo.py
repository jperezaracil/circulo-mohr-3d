"""
Página Streamlit — Pandeo de columnas.

Esta página solo se encarga de la UI: pinta los sliders, llama a los
módulos de cálculo (`pandeo_calculos`) y de gráfica (`pandeo_graficas`)
y muestra los resultados. Los catálogos de perfiles y materiales viven
en `pandeo_perfiles` y `pandeo_materiales` respectivamente.
"""

import sys
import os

# Asegurar que el directorio raíz está en el path para que las páginas
# (que viven en `paginas/`) puedan importar los módulos del root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from pandeo_materiales import MATERIALES
from pandeo_perfiles   import FAMILIAS
from pandeo_calculos   import (
    CONDICIONES,
    propiedades_seccion,
    radios_de_giro,
    esbeltez,
    esbeltez_limite,
    regimen_y_sigma,
)
from pandeo_graficas   import dibujar_columna, dibujar_curvas


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.title("Pandeo de columnas")
st.markdown(
    """
    Calcula la **carga crítica de pandeo** y la **tensión crítica** de
    una columna y compara las cuatro fórmulas clásicas (Euler, Johnson,
    Tetmajer, Rankine-Gordon) sobre el diagrama $\\sigma_{cr}$-$\\lambda$.

    Configura el material, la sección, la longitud y los apoyos en la
    barra lateral. Si introduces una carga aplicada $P$, se calcula
    también el coeficiente de seguridad.
    """
)


# ---------------------------------------------------------------------------
# Sidebar — entradas
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Material")
    mat_nombre = st.selectbox("Material", list(MATERIALES.keys()), index=1)
    mat = MATERIALES[mat_nombre]
    if mat_nombre == "Personalizado":
        E  = st.number_input("E (MPa)",
                             value=float(mat["E"]),
                             min_value=1000.0, step=1000.0)
        fy = st.number_input("σy (MPa)",
                             value=float(mat["fy"]),
                             min_value=1.0, step=10.0)
        tetmajer = None
    else:
        E  = mat["E"]
        fy = mat["fy"]
        tetmajer = mat["tetmajer"]
        st.caption(f"E = {E:.0f} MPa  ·  σy = {fy:.0f} MPa")

    st.divider()
    st.header("Sección")
    tipo = st.selectbox(
        "Tipo de sección",
        ["Catálogo", "Rectangular maciza", "Circular maciza",
         "Tubo circular", "Tubo rectangular"],
    )
    params = {}

    if tipo == "Catálogo":
        familia = st.selectbox("Familia", list(FAMILIAS.keys()))
        familia_dict = FAMILIAS[familia]
        params["nombre"] = st.selectbox("Perfil",
                                        list(familia_dict.keys()),
                                        index=min(5, len(familia_dict)-1))
    elif tipo == "Rectangular maciza":
        params["b"] = st.slider("b — anchura (mm)",  10, 500, 100, 5)
        params["h"] = st.slider("h — altura  (mm)",  10, 500, 200, 5)
    elif tipo == "Circular maciza":
        params["D"] = st.slider("Diámetro D (mm)",   10, 500, 100, 5)
    elif tipo == "Tubo circular":
        params["De"] = st.slider("De — diám. exterior (mm)",  20, 500, 100, 5)
        params["t"]  = st.slider("t  — espesor       (mm)",   1,  50,   5, 1)
    elif tipo == "Tubo rectangular":
        params["b"] = st.slider("b — anchura (mm)",  20, 500, 100, 5)
        params["h"] = st.slider("h — altura  (mm)",  20, 500, 200, 5)
        params["t"] = st.slider("t — espesor (mm)",   1,  50,   5, 1)

    st.divider()
    st.header("Geometría y apoyos")
    L    = st.slider("Longitud L (mm)", 100, 10_000, 3000, 50)
    cond = st.selectbox("Condiciones de contorno", list(CONDICIONES.keys()))
    K    = CONDICIONES[cond]["K"]

    st.divider()
    st.header("Carga aplicada (opcional)")
    P_aplicada = st.number_input(
        "P (kN)", value=0.0, min_value=0.0, step=10.0,
        help="Si > 0, se calcula el coeficiente de seguridad n = P_cr / P.",
    )


# ---------------------------------------------------------------------------
# Cálculos
# ---------------------------------------------------------------------------
A, Iy, Iz, h_sec, b_sec = propiedades_seccion(tipo, params)
r_y, r_z, r_min         = radios_de_giro(A, Iy, Iz)
Le, lam                 = esbeltez(L, K, r_min)
lam_lim                 = esbeltez_limite(E, fy)
sigma_cr, regimen       = regimen_y_sigma(lam, E, fy)
P_cr_kN                 = sigma_cr * A / 1000.0     # MPa·mm² = N → kN

n_seg = (P_cr_kN / P_aplicada) if P_aplicada > 0 else None


# ---------------------------------------------------------------------------
# Paneles gráficos
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 1.6])

with col1:
    fig1, ax1 = plt.subplots(figsize=(4.5, 7))
    dibujar_columna(ax1, K, L, n_seguridad=n_seg)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    dibujar_curvas(ax2, lam, sigma_cr, E, fy, tetmajer)
    st.pyplot(fig2)


# ---------------------------------------------------------------------------
# Tabla de resultados
# ---------------------------------------------------------------------------
st.subheader("Resultados")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Área A",          f"{A:.0f} mm²")
    st.metric("Inercia mín. Imin", f"{min(Iy, Iz):.2e} mm⁴")
with c2:
    st.metric("Radio giro mín. r_min", f"{r_min:.2f} mm")
    st.metric("Longitud pandeo Lₑ",    f"{Le:.0f} mm")
with c3:
    st.metric("Esbeltez λ",          f"{lam:.1f}")
    st.metric("Esbeltez límite λlim", f"{lam_lim:.1f}")
with c4:
    st.metric("Tensión crítica σcr", f"{sigma_cr:.1f} MPa")
    st.metric("Carga crítica Pcr",    f"{P_cr_kN:.2f} kN")

st.info(f"**Régimen**: {regimen}  ·  λ = {lam:.1f}, λlim = {lam_lim:.1f}")

if n_seg is not None:
    if n_seg >= 2.0:
        st.success(
            f"**Coeficiente de seguridad**  n = Pcr / P = "
            f"{P_cr_kN:.2f} / {P_aplicada:.2f} = **{n_seg:.2f}** ≥ 2 → seguro."
        )
    elif n_seg >= 1.0:
        st.warning(
            f"**Coeficiente de seguridad**  n = {n_seg:.2f}  "
            "(entre 1 y 2 — al límite, no es aceptable en diseño)."
        )
    else:
        st.error(
            f"**Coeficiente de seguridad**  n = {n_seg:.2f} < 1  "
            "→ la columna **PANDEA** bajo la carga aplicada."
        )


# ---------------------------------------------------------------------------
# Detalles teóricos plegables
# ---------------------------------------------------------------------------
with st.expander("Fórmulas y conceptos"):
    st.markdown(
        r"""
**Geometría y esbeltez**

- Área $A$ y momentos de inercia $I_y$, $I_z$ alrededor de los ejes
  fuerte y débil de la sección.
- **Radios de giro**:  $r_y = \sqrt{I_y/A}$,  $r_z = \sqrt{I_z/A}$.
  El pandeo se produce siempre alrededor del eje de menor inercia,
  así que se usa $r_{\min}$.
- **Longitud de pandeo** $L_e = K\cdot L$, donde $K$ depende de las
  condiciones de contorno:

  | Apoyos                       | K   |
  |---|---|
  | Biarticulada                 | 1.0 |
  | Biempotrada                  | 0.5 |
  | Empotrada-libre (voladizo)   | 2.0 |
  | Empotrada-articulada         | 0.7 |

- **Esbeltez** $\lambda = L_e / r_{\min}$.
- **Esbeltez límite** $\lambda_{\text{lím}} = \pi\sqrt{E/\sigma_y}$:
  separa el régimen elástico (Euler) del plástico (Johnson).

**Las cuatro fórmulas comparadas**

| Nombre | Fórmula | Rango usual de validez |
|---|---|---|
| Euler          | $\sigma_{cr} = \dfrac{\pi^2 E}{\lambda^2}$ | $\lambda > \lambda_{\text{lím}}$ |
| Johnson        | $\sigma_{cr} = \sigma_y - \dfrac{\sigma_y^2}{4\pi^2 E}\lambda^2$ | $0 \le \lambda \le \lambda_{\text{lím}}$ |
| Tetmajer       | $\sigma_{cr} = a - b\lambda$ (constantes empíricas) | $\lambda$ moderado |
| Rankine-Gordon | $\sigma_{cr} = \dfrac{\sigma_y}{1 + (\sigma_y/\pi^2 E)\lambda^2}$ | Todo $\lambda$ |

**Criterio de cálculo en esta app**

La tensión crítica que se reporta como "actual" usa Euler si
$\lambda \ge \lambda_{\text{lím}}$ y Johnson en caso contrario
(el régimen plástico mejor establecido teóricamente). Las otras dos
fórmulas se dibujan para comparación.

**Cargas y seguridad**

- Carga crítica  $P_{cr} = \sigma_{cr}\cdot A$.
- Coeficiente de seguridad  $n = P_{cr}/P_{\text{aplicada}}$.
  En estructuras metálicas se exige típicamente $n \ge 2$.
        """
    )
