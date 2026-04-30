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


# ---------------------------------------------------------------------------
# Problemas propuestos
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("Problemas propuestos")
st.caption(
    "Resuelve cada problema a mano, comprueba con la app, y solo después "
    "despliega la solución."
)


with st.expander("Problema 1 — Carga crítica de un IPE 200 esbelto", expanded=False):
    st.markdown(
        r"""
Calcula la carga crítica de pandeo de una columna **IPE 200** biarticulada,
de longitud $L = 4{,}0$ m, fabricada en acero **S275** ($E = 210\ \text{GPa}$,
$\sigma_y = 275\ \text{MPa}$).

Datos del IPE 200: $A = 28{,}5\ \text{cm}^2$,
$i_z = 2{,}24\ \text{cm}$ (radio de giro mínimo).

1. Calcula $\lambda$ y $\lambda_{\text{lím}}$. ¿Estamos en régimen Euler o Johnson?
2. Calcula $\sigma_{cr}$ y $P_{cr}$.
3. Verifica con la app.
"""
    )

with st.expander("Solución del Problema 1", expanded=False):
    st.markdown(
        r"""
**1)** Longitud de pandeo: $L_e = K\cdot L = 1{,}0 \cdot 4000 = 4000\ \text{mm}$.

$$\lambda = \frac{L_e}{i_z} = \frac{4000}{22{,}4} = 178{,}6$$

$$\lambda_{\text{lím}} = \pi\sqrt{\tfrac{E}{\sigma_y}} = \pi\sqrt{\tfrac{210\,000}{275}} = 86{,}9$$

Como $\lambda \gg \lambda_{\text{lím}}$ → **régimen Euler (elástico)**.

**2)** Tensión y carga críticas:

$$\sigma_{cr} = \frac{\pi^2 E}{\lambda^2} = \frac{\pi^2 \cdot 210\,000}{178{,}6^2} \approx 65{,}0\ \text{MPa}$$

$$P_{cr} = \sigma_{cr}\cdot A = 65{,}0 \cdot 2850 \approx 185\,000\ \text{N} \approx \boxed{185\ \text{kN}}$$

**3)** En la app: Material *S275*, sección *Catálogo IPE 200*, $L = 4000$ mm,
biarticulada. Reportará $P_{cr} \approx 185$ kN, régimen Euler.
"""
    )


with st.expander("Problema 2 — La misma columna, pero corta (régimen Johnson)",
                 expanded=False):
    st.markdown(
        r"""
Toma el mismo IPE 200 S275, biarticulado, pero ahora con longitud
$L = 1{,}0$ m.

1. Calcula $\lambda$. ¿Régimen Euler o Johnson?
2. Calcula $\sigma_{cr}$ por la fórmula de Johnson y $P_{cr}$.
3. ¿Qué obtendríamos si aplicáramos Euler "ingenuamente"? ¿Por qué eso es
   físicamente imposible?
"""
    )

with st.expander("Solución del Problema 2", expanded=False):
    st.markdown(
        r"""
**1)** $L_e = 1000$ mm; $\lambda = 1000 / 22{,}4 = 44{,}6$.

$\lambda_{\text{lím}} = 86{,}9$ (depende solo del material). Como
$\lambda < \lambda_{\text{lím}}$ → **régimen plástico (Johnson)**.

**2)** Fórmula de Johnson:

$$\sigma_{cr} = \sigma_y - \frac{\sigma_y^2}{4\pi^2 E}\,\lambda^2 = 275 - \frac{275^2}{4\pi^2\cdot 210\,000}\cdot 44{,}6^2$$

$$\sigma_{cr} = 275 - 17{,}99 \approx 257{,}0\ \text{MPa}$$

$$P_{cr} = 257{,}0 \cdot 2850 \approx 732{,}5\ \text{kN}$$

**3)** Si aplicáramos Euler:

$$\sigma_{cr}^{\text{Euler}} = \frac{\pi^2\cdot 210\,000}{44{,}6^2} \approx 1043\ \text{MPa}$$

Pero $\sigma_y = 275$ MPa: la columna **se plastificaría a compresión** mucho
antes de alcanzar esa tensión. La hipótesis de Euler (deformación elástica
hasta el inicio del pandeo) ya no es válida en columnas cortas. Por eso se
usa Johnson en $\lambda < \lambda_{\text{lím}}$.
"""
    )


with st.expander("Problema 3 — Comparación de las cuatro condiciones de apoyo",
                 expanded=False):
    st.markdown(
        r"""
Para el IPE 200 S275 de $L = 4{,}0$ m del Problema 1, calcula $P_{cr}$ con
las cuatro condiciones clásicas de contorno:

| Apoyos | $K$ | $L_e$ | $\lambda$ | $\sigma_{cr}$ (MPa) | $P_{cr}$ (kN) |
|---|---|---|---|---|---|
| Biarticulada | 1.0 | ? | ? | ? | ? |
| Biempotrada | 0.5 | ? | ? | ? | ? |
| Empotrada-libre (voladizo) | 2.0 | ? | ? | ? | ? |
| Empotrada-articulada | 0.7 | ? | ? | ? | ? |

¿Cuál da la **mayor** capacidad? ¿Cuál la **menor**? ¿Cuál es la relación
entre las dos extremas?
"""
    )

with st.expander("Solución del Problema 3", expanded=False):
    st.markdown(
        r"""
Con $i_z = 22{,}4$ mm, $A = 2850\ \text{mm}^2$, $E = 210\,000$ MPa,
$\lambda_{\text{lím}} = 86{,}9$:

| Apoyos | $K$ | $L_e$ (mm) | $\lambda$ | Régimen | $\sigma_{cr}$ (MPa) | $P_{cr}$ (kN) |
|---|---|---|---|---|---|---|
| Biarticulada | 1.0 | 4000 | 178{,}6 | Euler | 65{,}0  | **185** |
| Biempotrada  | 0.5 | 2000 |  89{,}3 | Euler | 260{,}0 | **741** |
| Empotrada-libre (voladizo) | 2.0 | 8000 | 357{,}1 | Euler | 16{,}3  | **46** |
| Empotrada-articulada | 0.7 | 2800 | 125{,}0 | Euler | 132{,}6 | **378** |

Mayor capacidad: **biempotrada** (741 kN).
Menor capacidad: **voladizo** (46 kN).

La relación entre la mejor y la peor es $741/46 \approx 16$. Es decir, una
columna empotrada por ambos extremos **soporta 16 veces más** que la misma
columna en voladizo.

Lección práctica: en estructuras, mejorar las condiciones de apoyo (empotrar
en lugar de articular) puede multiplicar la capacidad sin gastar más material.
Un voladizo es siempre el caso más desfavorable.
"""
    )


with st.expander("Problema 4 — Selección del IPE más pequeño",
                 expanded=False):
    st.markdown(
        r"""
Diseña una columna que soporte una carga axial $P = 200$ kN con un coeficiente
de seguridad $n = 2{,}5$, longitud $L = 3{,}0$ m, biarticulada, material **S275**.

Es decir: hay que encontrar el **IPE más pequeño** del catálogo tal que
$P_{cr} \ge n\cdot P = 500$ kN.

1. Procedimiento: para cada perfil del catálogo, calcula $r_{\min}$,
   $\lambda$, $\sigma_{cr}$ y $P_{cr}$, y comprueba si supera 500 kN.
2. ¿Qué IPE eliges?
3. Si cambiaras a acero S355 ($\sigma_y = 355$ MPa), ¿la respuesta cambiaría?
   Razónalo.
"""
    )

with st.expander("Solución del Problema 4", expanded=False):
    st.markdown(
        r"""
**1)** $L_e = 3000$ mm; $\lambda_{\text{lím}}\,(\text{S275}) = 86{,}9$.

Probamos varios perfiles (datos del catálogo):

| Perfil  | $A$ (mm²) | $i_z$ (mm) | $\lambda$ | Régimen | $\sigma_{cr}$ (MPa) | $P_{cr}$ (kN) | ¿OK? |
|---|---|---|---|---|---|---|---|
| IPE 200 | 2850 | 22{,}4 | 133{,}9 | Euler | 115{,}6 | 329 | NO |
| IPE 220 | 3340 | 24{,}8 | 121{,}1 | Euler | 141{,}4 | 472 | NO (casi) |
| **IPE 240** | **3910** | **26{,}9** | **111{,}4** | **Euler** | **167{,}1** | **653** | **SÍ** |

**2)** Elegimos **IPE 240** (el más pequeño que cumple).

Verificación: $n = P_{cr}/P = 653/200 = 3{,}27 \ge 2{,}5$ ✓.

**3)** Con S355: el cambio afecta a $\sigma_y$ pero **NO a $E$**, que es
prácticamente igual para todos los aceros estructurales.

$$\lambda_{\text{lím}}\,(\text{S355}) = \pi\sqrt{210\,000/355} = 76{,}4$$

El IPE 240 sigue teniendo $\lambda = 111{,}4 > 76{,}4$ → sigue en
**régimen Euler**. Y $\sigma_{cr}^{\text{Euler}} = \pi^2 E/\lambda^2$
**no depende de $\sigma_y$**: $P_{cr}$ es idéntico, 653 kN.

**Conclusión importante**: en columnas esbeltas (régimen Euler), aumentar
la calidad del acero **no aumenta la carga crítica de pandeo**. Lo único
que sirve es aumentar la inercia (perfil mayor) o reducir la longitud de
pandeo (mejorar apoyos). Es un error de diseño habitual.
"""
    )

