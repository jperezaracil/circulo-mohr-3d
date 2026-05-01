"""
Página Streamlit — Mohr 2D en la sección YZ.

Se asume que el eje longitudinal de la pieza es X y que el plano de
análisis es YZ (la sección transversal). Las componentes activas
del estado tensional son σy, σz, τyz.
"""

import sys
import os

# Permitir importar los módulos del root del proyecto.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from mohr_2d_calculos import (
    transformar_2d,
    principales_2d,
    tensiones_en_plano_2d,
)
from mohr_2d_graficas import dibujar_elemento_2d, dibujar_mohr_2d


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.title("Mohr 2D — Sección YZ")

st.markdown(
    r"""
Esta página analiza el estado tensional **plano** sobre la sección
transversal de una pieza cuyo eje longitudinal es $X$. El plano que
miramos es el $YZ$ y las componentes activas son $\sigma_y$, $\sigma_z$,
$\tau_{yz}$. Las cinco restantes se asumen nulas.

Modifica las tres componentes del tensor y el ángulo de giro $\theta$
del elemento en la barra lateral. Activa abajo la opción
**"Plano de corte"** para definir un plano oblicuo (recta en 2D) por
el ángulo $\alpha$ de su normal y ver el vector tracción descompuesto
en el plano.
"""
)


# ---------------------------------------------------------------------------
# Sidebar — entradas
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Tensor de tensiones (plano YZ)")
    sigma_y = st.slider("σy",  -100, 100,  60, 5)
    sigma_z = st.slider("σz",  -100, 100,  20, 5)
    tau_yz  = st.slider("τyz",  -80,  80,  30, 5)

    st.header("Rotación del elemento")
    theta = st.slider("θ — giro en plano YZ (°)", -90, 90, 0, 1)

    st.divider()
    preset = st.selectbox(
        "Casos didácticos (preset)",
        ["—",
         "Tracción uniaxial en Y",
         "Tracción uniaxial en Z",
         "Cortante puro YZ",
         "Estado biaxial (σy = σz)",
         "Compresión + cortante"]
    )

    if preset == "Tracción uniaxial en Y":
        sigma_y, sigma_z, tau_yz = 100, 0, 0
    elif preset == "Tracción uniaxial en Z":
        sigma_y, sigma_z, tau_yz = 0, 100, 0
    elif preset == "Cortante puro YZ":
        sigma_y, sigma_z, tau_yz = 0, 0, 50
    elif preset == "Estado biaxial (σy = σz)":
        sigma_y, sigma_z, tau_yz = 60, 60, 0
    elif preset == "Compresión + cortante":
        sigma_y, sigma_z, tau_yz = -60, 20, 30

    st.divider()
    st.header("Plano de corte (opcional)")
    activar_plano = st.checkbox(
        "Mostrar tensiones sobre un plano de inclinación dada",
        value=False,
        help="Define un plano (recta en 2D) en el plano YZ por el ángulo α "
             "de su normal con el eje y."
    )
    if activar_plano:
        alpha = st.slider("α — ángulo de la normal con el eje y (°)",
                          -90, 90, 30, 1)
    else:
        alpha = None


# ---------------------------------------------------------------------------
# Cálculos
# ---------------------------------------------------------------------------
syp, szp, typ        = transformar_2d(sigma_y, sigma_z, tau_yz, theta)
s1, s2, R, C, theta_p = principales_2d(sigma_y, sigma_z, tau_yz)

plano = (tensiones_en_plano_2d(sigma_y, sigma_z, tau_yz, alpha)
         if activar_plano else None)


# ---------------------------------------------------------------------------
# Paneles gráficos
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
dibujar_elemento_2d(ax1, sigma_y, sigma_z, tau_yz, theta, plano=plano)
dibujar_mohr_2d  (ax2, sigma_y, sigma_z, tau_yz, theta, plano=plano)
plt.tight_layout()
st.pyplot(fig)


# ---------------------------------------------------------------------------
# Resultados numéricos
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Tensor inicial")
    st.latex(rf"""
\begin{{pmatrix}}
{sigma_y:+.1f} & {tau_yz:+.1f} \\ {tau_yz:+.1f} & {sigma_z:+.1f}
\end{{pmatrix}}
""")
with col2:
    st.subheader(rf"Tensor girado (θ = {theta:.0f}°)")
    st.latex(rf"""
\begin{{pmatrix}}
{syp:+.1f} & {typ:+.1f} \\ {typ:+.1f} & {szp:+.1f}
\end{{pmatrix}}
""")
with col3:
    st.subheader("Tensiones principales")
    st.latex(rf"\sigma_1 = {s1:+.2f}")
    st.latex(rf"\sigma_2 = {s2:+.2f}")
    st.latex(rf"\theta_p = {theta_p:+.2f}^{{\circ}}")
    st.latex(rf"\tau_{{máx}} = {R:.2f}")


# ---------------------------------------------------------------------------
# Plano de corte: detalle numérico
# ---------------------------------------------------------------------------
if plano is not None:
    st.markdown("---")
    st.subheader("Tensiones sobre el plano de corte")

    n = plano["n"]
    t = plano["t"]

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Geometría del plano**")
        st.latex(rf"\vec{{n}} = ({n[0]:+.3f},\ {n[1]:+.3f})")
        st.caption(rf"(α = {alpha:.0f}° respecto al eje y)")
        st.markdown(r"**Vector tracción** $\vec{t} = \boldsymbol{\sigma}\cdot\vec{n}$ (en ejes y, z)")
        st.latex(rf"""
\vec{{t}} = \begin{{pmatrix}} {t[0]:+.3f} \\ {t[1]:+.3f} \end{{pmatrix}}
""")
    with cB:
        st.markdown("**Componentes en ejes locales del plano**")
        st.caption(
            "Los ejes locales son la normal $\\vec{n}$ y la tangente "
            "$\\vec{e}_t$ (perpendicular a $\\vec{n}$ en sentido antihorario)."
        )
        st.latex(rf"\sigma_n = \vec{{t}}\cdot\vec{{n}} = {plano['sigma_n']:+.3f}")
        st.latex(rf"\tau\ (\text{{tangencial}}) = {plano['tau_signed']:+.3f}")
        st.latex(rf"|\vec{{\tau}}| = {plano['tau_mag']:.3f}")

    st.warning(
        "**Cuidado con los ejes**: $\\sigma_n$ y $\\tau$ están expresados en "
        "los **ejes locales del plano** ($\\vec{n}, \\vec{e}_t$), no en los "
        "ejes globales $y, z$. La estrella magenta sobre el círculo de Mohr "
        "marca el punto $(\\sigma_n,\\, -\\tau)$ siguiendo la convención del "
        "diagrama (con $\\tau$ positivo hacia abajo cuando se aplica el signo "
        "tangencial)."
    )


# ---------------------------------------------------------------------------
# Explicación: convención de la rotación
# ---------------------------------------------------------------------------
with st.expander(
    "¿Por qué se mueven los puntos sobre el círculo cuando giro el elemento?"
):
    st.markdown(
        r"""
Cada uno de los puntos coloreados sobre el círculo corresponde a una
**cara del elemento girado** en el plano YZ:

- Punto **rojo** Y' — tensiones sobre la cara cuya normal es el eje
  $y'$: $(\sigma_{y'},\ -\tau_{y'z'})$.
- Punto **azul** Z' — tensiones sobre la cara cuya normal es el eje
  $z'$: $(\sigma_{z'},\ +\tau_{y'z'})$.

Ambos puntos están **diametralmente opuestos** sobre el círculo y
representan dos caras perpendiculares del mismo elemento.

Cuando giras el elemento un ángulo $\theta$ en sentido antihorario,
los dos puntos rotan **$2\theta$ también antihorario** sobre el
círculo (con la convención usada aquí, que plotea Y en
$(\sigma_y, -\tau_{yz})$). La razón del factor 2 es que las tensiones
varían como $\cos(2\theta)$ y $\sin(2\theta)$ en las fórmulas de
transformación.

**No se mueven** ni el círculo, ni las tensiones principales
$\sigma_1, \sigma_2$, ni el centro $C$, ni el radio $R$. Eso es porque
**el estado tensional es el mismo**: lo que cambia es la orientación de
las caras sobre las que medimos las tensiones.

Cuando los dos puntos caen sobre el eje $\sigma$ (con $\tau = 0$), el
elemento está orientado según las **direcciones principales**: las
caras solo soportan tensión normal, sin cortante.
"""
    )


# ---------------------------------------------------------------------------
# Problemas propuestos
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("Problemas propuestos")
st.caption(
    "Resuelve cada problema a mano, comprueba con la app, y solo "
    "después despliega la solución."
)


with st.expander("Problema 1 — Tensiones principales en la sección YZ",
                 expanded=False):
    st.markdown(
        r"""
En la sección $YZ$ de una pieza cilíndrica con eje longitudinal $X$,
el estado tensional es:

$$\sigma_y = 60\ \text{MPa},\quad \sigma_z = 20\ \text{MPa},\quad \tau_{yz} = 30\ \text{MPa}.$$

Calcula:

1. Las **tensiones principales** $\sigma_1$ y $\sigma_2$.
2. El **ángulo principal** $\theta_p$ que orienta $\sigma_1$
   respecto al eje $y$.
3. La **tensión cortante máxima** en el plano.
4. Verifica con la app.
"""
    )

with st.expander("Solución del Problema 1", expanded=False):
    st.markdown(
        r"""
Centro y radio del círculo:

$$C = \frac{\sigma_y + \sigma_z}{2} = 40\ \text{MPa}$$

$$R = \sqrt{\left(\tfrac{\sigma_y - \sigma_z}{2}\right)^2 + \tau_{yz}^2}
   = \sqrt{20^2 + 30^2} = \sqrt{1300} \approx 36{,}06\ \text{MPa}$$

**1) Principales:**

$$\sigma_1 = C + R \approx 76{,}06\ \text{MPa},\qquad
  \sigma_2 = C - R \approx 3{,}94\ \text{MPa}$$

**2) Ángulo principal:**

$$\tan(2\theta_p) = \frac{2\tau_{yz}}{\sigma_y - \sigma_z}
                  = \frac{60}{40} = 1{,}5
\;\Rightarrow\; 2\theta_p \approx 56{,}31^{\circ}
\;\Rightarrow\; \theta_p \approx 28{,}15^{\circ}$$

**3) Cortante máxima en el plano:**

$$\tau_{\max} = R \approx 36{,}06\ \text{MPa}$$

aparece a $\theta_p + 45^{\circ} \approx 73{,}15^{\circ}$ del eje $y$.
"""
    )


with st.expander("Problema 2 — Tensiones sobre un plano oblicuo",
                 expanded=False):
    st.markdown(
        r"""
Para el mismo estado tensional del Problema 1
($\sigma_y = 60,\ \sigma_z = 20,\ \tau_{yz} = 30$ MPa), calcula la
tensión normal y la cortante que actúan sobre un plano cuya normal
forma un ángulo $\alpha = 30^{\circ}$ con el eje $y$.

Verifica el resultado con la app activando *"Plano de corte"* con
$\alpha = 30^{\circ}$.
"""
    )

with st.expander("Solución del Problema 2", expanded=False):
    st.markdown(
        r"""
Las fórmulas de transformación dan, sustituyendo $\alpha = 30^{\circ}$
($\cos^2 30 = 0{,}75$, $\sin^2 30 = 0{,}25$, $\sin 30\cos 30 = \tfrac{\sqrt 3}{4}$):

$$\sigma_n = \sigma_y\cos^2\alpha + \sigma_z\sin^2\alpha + 2\tau_{yz}\sin\alpha\cos\alpha$$

$$\sigma_n = 60\cdot 0{,}75 + 20\cdot 0{,}25 + 2\cdot 30\cdot 0{,}433
           = 45 + 5 + 25{,}98 \approx 75{,}98\ \text{MPa}$$

$$\tau = -(\sigma_y - \sigma_z)\sin\alpha\cos\alpha + \tau_{yz}(\cos^2\alpha - \sin^2\alpha)$$

$$\tau = -40\cdot 0{,}433 + 30\cdot 0{,}5 \approx -17{,}32 + 15 = -2{,}32\ \text{MPa}$$

Es decir, $\sigma_n \approx 76{,}0$ MPa y $\tau \approx -2{,}3$ MPa.
La estrella magenta cae prácticamente sobre el extremo derecho del
círculo, casi en $\sigma_1$: efectivamente, $\alpha = 30^{\circ}$ está
muy cerca del ángulo principal $\theta_p \approx 28{,}15^{\circ}$.
"""
    )


with st.expander("Problema 3 — Cortante puro y direcciones principales",
                 expanded=False):
    st.markdown(
        r"""
Considera un estado de cortante puro en YZ: $\sigma_y = \sigma_z = 0$,
$\tau_{yz} = 50$ MPa.

1. ¿Qué tensiones principales tiene este estado?
2. ¿Para qué orientación $\theta$ del elemento desaparece la tensión
   cortante? ¿Qué tensiones aparecen entonces sobre las caras?
3. Comprueba con la app: preset *"Cortante puro YZ"* y mueve $\theta$.
"""
    )

with st.expander("Solución del Problema 3", expanded=False):
    st.markdown(
        r"""
**1)** $C = 0$, $R = 50$. Las principales son
$\sigma_1 = +50$, $\sigma_2 = -50$ MPa.

**2)** El cortante se anula en las direcciones principales:
$\tan(2\theta_p) = 2\tau_{yz}/(\sigma_y - \sigma_z) \to \infty$, así
que $2\theta_p = 90^{\circ}$ y $\theta_p = 45^{\circ}$.

Al girar el elemento $45^{\circ}$, las caras pasan a ver:
$\sigma_{y'} = +50$, $\sigma_{z'} = -50$, $\tau_{y'z'} = 0$. Es
**tracción + compresión cruzadas** sobre las dos caras.

Esto explica por qué un eje sometido a torsión rompe a $\pm 45^{\circ}$
respecto a su eje longitudinal en materiales frágiles (rotura por
tracción) — los planos a $45^{\circ}$ son los que ven la mayor
tensión normal.
"""
    )


with st.expander("Problema 4 — Diseño con criterio de Tresca",
                 expanded=False):
    st.markdown(
        r"""
Un punto de un sólido tiene $\sigma_y = 80$, $\sigma_z = -20$,
$\tau_{yz} = 40$ MPa. El material es un acero con límite elástico
$\sigma_y^{\,\text{mat}} = 250$ MPa.

Aplicando el criterio de Tresca (cortante máxima) en 2D, calcula:

1. Las tensiones principales en el plano.
2. La cortante máxima 2D.
3. El coeficiente de seguridad estático $n = \sigma_y^{\,\text{mat}}/(2\tau_{\max})$.
"""
    )

with st.expander("Solución del Problema 4", expanded=False):
    st.markdown(
        r"""
**1)** $C = (80 - 20)/2 = 30$, $R = \sqrt{50^2 + 40^2}
       = \sqrt{4100} \approx 64{,}03$ MPa.

$$\sigma_1 \approx 94{,}0\ \text{MPa},\qquad
  \sigma_2 \approx -34{,}0\ \text{MPa}$$

**2)** Cortante máxima en el plano: $\tau_{\max,\,2D} = R \approx 64{,}0$ MPa.

> *Nota*: en 3D, como en este problema $\sigma_3 = \sigma_x = 0$ y
> $\sigma_2 \approx -34$, la cortante máxima 3D es realmente
> $(\sigma_1 - \sigma_2)/2 \approx 64{,}0$ MPa (igual que la 2D porque
> $\sigma_3$ está entre $\sigma_2$ y $\sigma_1$).

**3)** Tresca: rotura cuando $2\tau_{\max} = \sigma_y^{\,\text{mat}}$.

$$n = \frac{\sigma_y^{\,\text{mat}}}{2\tau_{\max}}
    = \frac{250}{2\cdot 64{,}0} \approx 1{,}95$$

Está por debajo del valor recomendado ($n \ge 2$), por lo que la pieza
está al límite y conviene rediseñarla.
"""
    )
