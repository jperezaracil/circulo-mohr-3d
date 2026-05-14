"""
Página Streamlit — Ejercicios interactivos del círculo de Mohr.

Dos bloques:

  1) Explorador interactivo: el alumno mueve un único punto sobre el
     círculo (mediante el ángulo α de la normal del plano) y ve, en
     tiempo real, qué representa ese punto: la orientación del plano,
     las tensiones (σ_n, τ) que actúan sobre él y los puntos
     singulares del círculo (caras y, z, direcciones principales,
     cortante máxima).

  2) Serie de ejercicios resueltos: 6 ejercicios pedagógicos con
     enunciado y solución detallada bajo expanders.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# Importar módulos del root del proyecto
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import estilo  # noqa: F401  (paleta clara)
from estilo import PALETA
from mohr_2d_calculos import (
    transformar_2d,
    principales_2d,
    tensiones_en_plano_2d,
)


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.title("Ejercicios del círculo de Mohr")
st.markdown(
    """
Esta página tiene dos partes:

1. **Explorador interactivo** — mueve un punto por el círculo de Mohr y
   observa, en tiempo real, qué representa físicamente: la cara de un
   elemento orientado a cierto ángulo, una dirección principal, la
   cortante máxima…
2. **Ejercicios resueltos** — una serie progresiva de problemas
   pensados para resolver a mano. Cada uno tiene su solución oculta:
   intenta resolverlo antes de mirarla.
"""
)

st.markdown("---")


# ---------------------------------------------------------------------------
# 1) Explorador interactivo
# ---------------------------------------------------------------------------
st.header("1. Explorador interactivo")
st.markdown(
    r"""
Define un estado tensional 2D y desliza el ángulo $\alpha$ de la normal
del plano. El **punto naranja** sobre el círculo te dice las tensiones
$(\sigma_n, \tau)$ que ve ese plano. La animación deja claro un hecho
central del círculo de Mohr: si giras el plano un ángulo $\alpha$ en
el espacio físico, el punto recorre $2\alpha$ sobre el círculo.
"""
)

cE1, cE2 = st.columns([1, 1])
with cE1:
    st.markdown("**Estado tensional**")
    sy_e = st.slider("σy",  -100, 100, 70, 5, key="ex_sy")
    sz_e = st.slider("σz",  -100, 100, 20, 5, key="ex_sz")
    tyz_e = st.slider("τyz", -80,  80, 30, 5, key="ex_tyz")
with cE2:
    st.markdown("**Orientación del plano**")
    alpha_e = st.slider(
        "α — ángulo de la normal n̂ con el eje y (°)",
        -90, 90, 0, 1, key="ex_alpha",
    )
    posicion = st.radio(
        "Posiciones singulares",
        ["—", "Cara y (α=0°)", "Cara z (α=90°)",
         "Dirección principal σ₁", "Dirección principal σ₂",
         "Cortante máxima (+)", "Cortante máxima (−)"],
        index=0, key="ex_posicion",
    )

# Resolver presets — sobreescriben el slider de α
s1_e, s2_e, R_e, C_e, theta_p_e = principales_2d(sy_e, sz_e, tyz_e)
if posicion == "Cara y (α=0°)":
    alpha_e = 0
elif posicion == "Cara z (α=90°)":
    alpha_e = 90
elif posicion == "Dirección principal σ₁":
    alpha_e = float(theta_p_e)
elif posicion == "Dirección principal σ₂":
    alpha_e = float(theta_p_e + 90.0)
    if alpha_e > 90.0:
        alpha_e -= 180.0
elif posicion == "Cortante máxima (+)":
    alpha_e = float(theta_p_e + 45.0)
    if alpha_e > 90.0:
        alpha_e -= 180.0
elif posicion == "Cortante máxima (−)":
    alpha_e = float(theta_p_e - 45.0)
    if alpha_e <= -90.0:
        alpha_e += 180.0


# --- Cálculo de las tensiones en el plano α ---
pl = tensiones_en_plano_2d(sy_e, sz_e, tyz_e, alpha_e)
sigma_n_alpha = pl["sigma_n"]
tau_signed_alpha = pl["tau_signed"]   # τ con signo en ejes locales
# El diagrama de Mohr usa convención: punto Y' en (σ_y', −τ_y'z'). Para
# el ángulo α de la normal de la cara y', el par graficado es:
sigma_q_plot = sigma_n_alpha
tau_q_plot   = -tau_signed_alpha   # τ "tal y como se ve en el diagrama"


# --- Dibujo ---
def _flecha(ax, x0, y0, dx, dy, color, lw=2.0, ls='-'):
    ax.annotate('', xy=(x0+dx, y0+dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color,
                                lw=lw, mutation_scale=14, linestyle=ls))


def _dibujar_explorador(ax_elem, ax_mohr):
    a = np.deg2rad(alpha_e)
    n    = np.array([np.cos(a),  np.sin(a)])
    tang = np.array([-np.sin(a), np.cos(a)])

    # --- Panel izquierdo: elemento + plano de corte ---
    L = 1.0
    v0 = np.array([[L, L], [-L, L], [-L, -L], [L, -L], [L, L]])
    ax_elem.plot(v0[:, 0], v0[:, 1], '-', color=PALETA["borde"], lw=1.0)
    ax_elem.fill(v0[:-1, 0], v0[:-1, 1], color=PALETA["celeste"], alpha=0.35)

    # Línea del plano (recta perpendicular a n por el origen)
    size = 1.9 * L
    p1 = -size * tang
    p2 =  size * tang
    ax_elem.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color=PALETA["naranja"], lw=2.4, alpha=0.85,
                 label='Plano de corte')

    # Normal del plano
    _flecha(ax_elem, 0, 0, *(1.6 * n), color=PALETA["verde"], lw=1.8)
    ax_elem.text(*(1.78 * n), r'$\vec{n}$',
                 color=PALETA["verde"], fontsize=12, fontweight='bold')

    # Vector tracción y descomposición (σ_n n + τ tang)
    smax = max(abs(sy_e), abs(sz_e), abs(tyz_e), 1e-9)
    k = 0.014  # escala fija razonable para un cuadrado de lado 2
    k = 0.8 / smax
    t_vec = pl["t"]
    _flecha(ax_elem, 0, 0, *(t_vec * k), color=PALETA["morado"], lw=2.5)
    ax_elem.text(*(t_vec * k * 1.1), r'$\vec{t}$',
                 color=PALETA["morado"], fontsize=12, fontweight='bold')
    _flecha(ax_elem, 0, 0, *(sigma_n_alpha * k * n),
            color=PALETA["rojo"], lw=1.4, ls='--')
    if abs(tau_signed_alpha) > 1e-6 * smax:
        _flecha(ax_elem, 0, 0,
                *(tau_signed_alpha * k * tang),
                color=PALETA["azul"], lw=1.4, ls='--')

    # Ejes globales y, z
    _flecha(ax_elem, 0, 0, 2.6, 0, color=PALETA["texto_suave"], lw=1.0)
    _flecha(ax_elem, 0, 0, 0, 2.6, color=PALETA["texto_suave"], lw=1.0)
    ax_elem.text(2.7, 0.05, 'y', color=PALETA["texto_suave"])
    ax_elem.text(0.05, 2.7, 'z', color=PALETA["texto_suave"])

    # Arco indicador de α
    if abs(alpha_e) > 0.3:
        arc = mpatches.Arc((0, 0), 0.75, 0.75,
                           theta1=min(0.0, alpha_e),
                           theta2=max(0.0, alpha_e),
                           color=PALETA["verde"], lw=1.5)
        ax_elem.add_patch(arc)
        ax_elem.text(0.55*np.cos(a/2), 0.55*np.sin(a/2),
                     rf'$\alpha={alpha_e:.0f}°$',
                     color=PALETA["verde"], fontsize=10)

    ax_elem.set_xlim(-3, 3)
    ax_elem.set_ylim(-3, 3)
    ax_elem.set_aspect('equal')
    ax_elem.set_title('Elemento físico — plano (naranja) con normal n̂')
    ax_elem.axhline(0, color=PALETA["rejilla"], lw=0.6, zorder=0)
    ax_elem.axvline(0, color=PALETA["rejilla"], lw=0.6, zorder=0)

    # --- Panel derecho: círculo de Mohr con un único punto ---
    th_circ = np.linspace(0, 2*np.pi, 200)
    ax_mohr.plot(C_e + R_e*np.cos(th_circ), R_e*np.sin(th_circ),
                 '-', color=PALETA["texto"], lw=1.4)
    ax_mohr.plot(C_e, 0, '+', color=PALETA["texto"], markersize=10)

    # Tensiones principales
    ax_mohr.plot([s1_e, s2_e], [0, 0], 'o',
                 color=PALETA["verde"], markersize=9, zorder=4)
    ax_mohr.text(s1_e, -0.08*R_e - 1, rf'$\sigma_1={s1_e:.1f}$',
                 color=PALETA["verde"], ha='center', va='top', fontsize=10)
    ax_mohr.text(s2_e,  0.08*R_e + 1, rf'$\sigma_2={s2_e:.1f}$',
                 color=PALETA["verde"], ha='center', va='bottom', fontsize=10)

    # Puntos especiales (gris claro)
    # Caras y, z iniciales
    ax_mohr.plot(sy_e, -tyz_e, 'o', color=PALETA["texto_suave"],
                 markersize=6, alpha=0.7)
    ax_mohr.text(sy_e, -tyz_e, '   y (α=0°)',
                 color=PALETA["texto_suave"], fontsize=8, va='center')
    ax_mohr.plot(sz_e,  tyz_e, 'o', color=PALETA["texto_suave"],
                 markersize=6, alpha=0.7)
    ax_mohr.text(sz_e,  tyz_e, '   z (α=90°)',
                 color=PALETA["texto_suave"], fontsize=8, va='center')

    # Cortante máxima
    ax_mohr.plot([C_e, C_e], [R_e, -R_e], 'o',
                 color=PALETA["magenta"], markersize=6, alpha=0.7)
    ax_mohr.text(C_e + 0.04*R_e, R_e, rf'  $\tau_{{máx}}={R_e:.1f}$',
                 color=PALETA["magenta"], fontsize=8, va='center')

    # Punto actual: corresponde al plano de normal α (cara y' en α)
    ax_mohr.plot(sigma_q_plot, tau_q_plot, marker='D',
                 color=PALETA["naranja"], markersize=14,
                 markeredgecolor=PALETA["texto"], markeredgewidth=0.8,
                 zorder=6, label=f'Plano α={alpha_e:.0f}°')
    ax_mohr.text(sigma_q_plot, tau_q_plot,
                 f"  ({sigma_q_plot:.1f}, {tau_q_plot:+.1f})",
                 color=PALETA["naranja"], fontsize=10,
                 fontweight='bold', va='center')

    # Radio desde el centro al punto, para visualizar 2α
    ax_mohr.plot([C_e, sigma_q_plot], [0, tau_q_plot],
                 '-', color=PALETA["naranja"], lw=1.2, alpha=0.6)

    # Arco indicador de 2α en el círculo (desde la cara y)
    ang_y   = np.degrees(np.arctan2(-tyz_e, (sy_e - sz_e)/2.0))
    ang_now = np.degrees(np.arctan2(tau_q_plot, sigma_q_plot - C_e))
    if abs(ang_now - ang_y) > 1.0:
        rad_arc = 0.30 * R_e
        arc = mpatches.Arc((C_e, 0), 2*rad_arc, 2*rad_arc,
                           theta1=min(ang_y, ang_now),
                           theta2=max(ang_y, ang_now),
                           color=PALETA["naranja"], lw=1.5, alpha=0.7)
        ax_mohr.add_patch(arc)
        ang_med = np.deg2rad((ang_y + ang_now)/2.0)
        ax_mohr.text(C_e + 1.2*rad_arc*np.cos(ang_med),
                     0 + 1.2*rad_arc*np.sin(ang_med),
                     rf'$2\alpha={2*alpha_e:.0f}°$',
                     color=PALETA["naranja"], fontsize=9, ha='center')

    margen = 0.30 * R_e + 1.0
    ax_mohr.set_xlim(min(s2_e, 0) - margen, max(s1_e, 0) + margen)
    ax_mohr.set_ylim(-R_e - margen, R_e + margen)
    ax_mohr.set_aspect('equal')
    ax_mohr.set_xlabel(r'$\sigma$')
    ax_mohr.set_ylabel(r'$\tau$')
    ax_mohr.axhline(0, color=PALETA["rejilla"], lw=0.6, zorder=0)
    ax_mohr.axvline(0, color=PALETA["rejilla"], lw=0.6, zorder=0)
    ax_mohr.set_title(rf'Círculo de Mohr  ·  $C={C_e:.1f}$  ·  $R={R_e:.1f}$')


fig_e = plt.figure(figsize=(12, 5.5))
ax_elem = fig_e.add_subplot(1, 2, 1)
ax_mohr = fig_e.add_subplot(1, 2, 2)
_dibujar_explorador(ax_elem, ax_mohr)
plt.tight_layout()
st.pyplot(fig_e)


# --- Texto interpretativo del punto actual ---
def _interpretacion(alpha, theta_p, R, s1, s2, sigma_n, tau):
    """Devuelve un texto breve sobre qué representa el punto actual."""
    tol = 1.5
    # Distancia angular a posiciones notables (modulo 180°)
    def diff(a, b):
        d = (a - b + 90.0) % 180.0 - 90.0
        return abs(d)

    if diff(alpha, theta_p) < tol:
        return ("Estás sobre la **dirección principal σ₁**: el plano "
                "solo soporta tensión normal (τ = 0) y su valor coincide "
                "con la mayor tensión principal.")
    if diff(alpha, theta_p + 90.0) < tol:
        return ("Estás sobre la **dirección principal σ₂**: el plano "
                "solo soporta tensión normal y su valor coincide con la "
                "menor tensión principal.")
    if diff(alpha, theta_p + 45.0) < tol or diff(alpha, theta_p - 45.0) < tol:
        return ("Estás en un plano de **cortante máxima**: |τ| = R, el "
                "radio del círculo. La σ del plano coincide con el "
                "centro C = (σ₁+σ₂)/2.")
    if diff(alpha, 0.0) < tol:
        return ("Estás sobre la **cara y** del elemento original "
                "(α = 0°). La σ es σy y la τ es −τyz (convención del "
                "diagrama).")
    if diff(alpha, 90.0) < tol:
        return ("Estás sobre la **cara z** del elemento original "
                "(α = 90°). El punto está diametralmente opuesto al de "
                "la cara y porque ambas caras son perpendiculares.")
    return ("Estás en un plano **genérico**: la cara orientada a este "
            f"ángulo ve σ_n = {sigma_n:.1f} y τ = {tau:+.1f}. Observa "
            "cómo al cambiar α en 1° el punto avanza 2° sobre el círculo.")


st.info(
    f"**¿Qué representa el punto naranja?** " +
    _interpretacion(alpha_e, theta_p_e, R_e, s1_e, s2_e,
                    sigma_n_alpha, -tau_signed_alpha)
)

cR1, cR2, cR3 = st.columns(3)
with cR1:
    st.metric("σ_n del plano",   f"{sigma_n_alpha:+.2f}")
    st.metric("τ del plano (signo en ejes locales)",
              f"{tau_signed_alpha:+.2f}")
with cR2:
    st.metric("σ₁", f"{s1_e:+.2f}")
    st.metric("σ₂", f"{s2_e:+.2f}")
with cR3:
    st.metric("C (centro)", f"{C_e:+.2f}")
    st.metric("R (radio)",  f"{R_e:.2f}")

with st.expander("Cómo leer el explorador"):
    st.markdown(
        r"""
- El **rombo naranja** del círculo es **la cara y'** del elemento
  cuando lo giras un ángulo $\alpha$ — es decir, el plano cuya normal
  forma un ángulo $\alpha$ con el eje $y$ del estado original.
- Si avanzas $\alpha$ un grado en el panel izquierdo, el punto
  recorre **dos grados** en el panel derecho — esa es la conocida
  duplicación angular del círculo de Mohr.
- Cuando $\alpha = \theta_p$ el punto cae sobre el eje $\sigma$ a la
  derecha: ese plano es la **dirección principal $\sigma_1$**.
  Girando $\alpha$ otros $45°$ caes sobre el plano de **cortante
  máxima** (extremo superior o inferior del círculo).
- Los puntos grises del círculo te recuerdan, además, las posiciones
  notables: caras $y$ y $z$ del estado inicial y los dos puntos de
  $\tau$ máximo.
        """
    )


# ---------------------------------------------------------------------------
# 2) Ejercicios con solución
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("2. Ejercicios con solución")
st.caption(
    "Resuelve a mano, comprueba moviendo el punto en el explorador, y "
    "solo después abre el desplegable con la solución detallada."
)


# Helper para enunciados/soluciones
def _ejercicio(titulo, enunciado, solucion):
    with st.expander(f"📝  {titulo}", expanded=False):
        st.markdown(enunciado)
    with st.expander(f"✅  Solución — {titulo}", expanded=False):
        st.markdown(solucion)


_ejercicio(
    "Ejercicio 1 — Tensiones principales clásicas",
    r"""
Un punto está sometido al estado tensional:

$$\sigma_y = 70\ \text{MPa},\quad \sigma_z = 20\ \text{MPa},\quad \tau_{yz} = 30\ \text{MPa}.$$

Calcula:

1. El centro $C$ y radio $R$ del círculo de Mohr.
2. Las tensiones principales $\sigma_1$ y $\sigma_2$.
3. El ángulo $\theta_p$ que orienta $\sigma_1$ respecto al eje $y$.
4. La cortante máxima en el plano.

> *Pista*: usa $C=(\sigma_y+\sigma_z)/2$ y
> $R=\sqrt{((\sigma_y-\sigma_z)/2)^2 + \tau_{yz}^2}$.
""",
    r"""
**Centro y radio:**

$$C = \frac{70 + 20}{2} = 45\ \text{MPa}$$

$$R = \sqrt{\left(\tfrac{70-20}{2}\right)^2 + 30^2} = \sqrt{25^2 + 30^2} = \sqrt{1525} \approx 39{,}05\ \text{MPa}$$

**Tensiones principales:**

$$\sigma_1 = C + R \approx 84{,}05\ \text{MPa},\qquad \sigma_2 = C - R \approx 5{,}95\ \text{MPa}$$

**Ángulo principal:**

$$\tan(2\theta_p) = \frac{2\tau_{yz}}{\sigma_y-\sigma_z} = \frac{60}{50} = 1{,}2 \;\Rightarrow\; 2\theta_p \approx 50{,}19^{\circ} \;\Rightarrow\; \theta_p \approx 25{,}1^{\circ}$$

**Cortante máxima en el plano**: $\tau_{\max} = R \approx 39{,}1\ \text{MPa}$,
en un plano a $\theta_p + 45^{\circ} \approx 70{,}1^{\circ}$ del eje $y$.

> *Verificación en el explorador*: pon $\sigma_y=70$, $\sigma_z=20$,
> $\tau_{yz}=30$ y selecciona "Dirección principal $\sigma_1$": el
> punto naranja salta a $\sigma \approx 84$ con $\tau=0$.
""",
)


_ejercicio(
    "Ejercicio 2 — Tensiones sobre un plano dado",
    r"""
Para el estado tensional del ejercicio 1
($\sigma_y=70,\ \sigma_z=20,\ \tau_{yz}=30$ MPa), calcula la tensión
normal y la cortante que actúan sobre un plano cuya normal forma un
ángulo $\alpha = 30^{\circ}$ con el eje $y$.

Después compara con la situación $\alpha = -60^{\circ}$. ¿Por qué tiene
sentido lo que observas?
""",
    r"""
Aplicando las fórmulas de transformación con $\alpha = 30^{\circ}$:

$$\sigma_n = \tfrac{\sigma_y+\sigma_z}{2} + \tfrac{\sigma_y-\sigma_z}{2}\cos 2\alpha + \tau_{yz}\sin 2\alpha = 45 + 25\cos 60^{\circ} + 30\sin 60^{\circ}$$

$$\sigma_n = 45 + 12{,}5 + 25{,}98 \approx 83{,}48\ \text{MPa}$$

$$\tau = -\tfrac{\sigma_y-\sigma_z}{2}\sin 2\alpha + \tau_{yz}\cos 2\alpha = -25\sin 60^{\circ} + 30\cos 60^{\circ}$$

$$\tau = -21{,}65 + 15 = -6{,}65\ \text{MPa}$$

Con $\alpha = -60^{\circ}$: $\cos(-120^{\circ}) = -0{,}5$,
$\sin(-120^{\circ}) = -0{,}866$:

$$\sigma_n = 45 - 12{,}5 - 25{,}98 \approx 6{,}52\ \text{MPa}$$

$$\tau = -25\cdot(-0{,}866) + 30\cdot(-0{,}5) = 21{,}65 - 15 = 6{,}65\ \text{MPa}$$

Los dos planos están **perpendiculares entre sí** ($30^{\circ}$ y
$-60^{\circ}$ se diferencian en $90^{\circ}$) y, en el círculo, sus
puntos son **diametralmente opuestos**: ambos cumplen
$\sigma_n^{(30)} + \sigma_n^{(-60)} \approx 90 = 2C$ y
$|\tau^{(30)}| = |\tau^{(-60)}|$ (con signos opuestos en ejes locales,
o iguales en la convención del diagrama).
""",
)


_ejercicio(
    "Ejercicio 3 — Estado de cortante puro",
    r"""
Un eje cilíndrico sometido a torsión pura presenta, sobre la superficie,
el estado:

$$\sigma_y = 0,\quad \sigma_z = 0,\quad \tau_{yz} = 60\ \text{MPa}.$$

1. ¿Qué centro y radio tiene su círculo de Mohr?
2. ¿Cuáles son las tensiones principales?
3. ¿En qué orientación $\alpha$ desaparece la cortante?
4. ¿Por qué los ejes de torsión rompen a $\pm 45^{\circ}$ en
   materiales frágiles?
""",
    r"""
**1)** $C = (0+0)/2 = 0$, $R = \sqrt{0 + 60^2} = 60$ MPa.

**2)** $\sigma_1 = +60$ MPa, $\sigma_2 = -60$ MPa.

**3)** $\tan(2\theta_p) = 2\tau_{yz}/(\sigma_y - \sigma_z) \to \infty$,
así que $2\theta_p = 90^{\circ}$ y $\theta_p = 45^{\circ}$. En las
direcciones $\alpha = 45^{\circ}$ y $\alpha = -45^{\circ}$ la cortante
en ejes locales se anula y las caras del elemento ven solo tensión
normal: tracción $+60$ MPa en $\alpha = 45^{\circ}$ y compresión $-60$
MPa en $\alpha = -45^{\circ}$.

**4)** Como los materiales frágiles (hormigón, hierro fundido, tiza)
rompen por tracción mucho antes que por cortante, las grietas se
abren **perpendiculares a la dirección de máxima tracción**: en este
estado, perpendiculares a $\alpha = 45^{\circ}$. El resultado es la
**rotura helicoidal a $45^{\circ}$** característica de los ejes
torsionados.

> *Verificación en el explorador*: $\sigma_y=\sigma_z=0$,
> $\tau_{yz}=60$. Selecciona "Dirección principal $\sigma_1$" y
> verás $\alpha = 45^{\circ}$, $\sigma = +60$, $\tau = 0$.
""",
)


_ejercicio(
    "Ejercicio 4 — Encontrar el plano que da una σ_n dada",
    r"""
Para el estado $\sigma_y = 80$, $\sigma_z = 20$, $\tau_{yz} = 0$,
encuentra **todos** los ángulos $\alpha$ (con $-90^{\circ}<\alpha\le 90^{\circ}$)
para los que la cara $y'$ vea $\sigma_n = 60$ MPa.

¿Cuánto vale $\tau$ en esos planos? ¿Es único el resultado?
""",
    r"""
Usando $\sigma_n = C + R\cos(2\alpha + \beta)$ con $C = 50$,
$R = 30$, y aquí $\beta = 0$ porque $\tau_{yz} = 0$:

$$60 = 50 + 30\cos(2\alpha) \;\Longrightarrow\; \cos(2\alpha) = \tfrac{1}{3}$$

$$2\alpha = \pm\arccos(\tfrac{1}{3}) \approx \pm 70{,}53^{\circ} \;\Longrightarrow\; \alpha \approx \pm 35{,}26^{\circ}$$

Para la cortante, $\tau_{\text{signed}} = -\tfrac{\sigma_y-\sigma_z}{2}\sin 2\alpha
+ \tau_{yz}\cos 2\alpha = -30\sin(\pm 70{,}53^{\circ})$:

$$\tau \approx \mp 28{,}28\ \text{MPa}$$

**El resultado no es único**: hay dos planos físicamente distintos que
cumplen $\sigma_n = 60$, y cada uno tiene $|\tau| \approx 28{,}3$ MPa
con signos opuestos. En el círculo, ambos puntos están sobre la recta
vertical $\sigma = 60$ y son simétricos respecto al eje $\sigma$.
""",
)


_ejercicio(
    "Ejercicio 5 — Diseño con criterio de Tresca",
    r"""
Un punto crítico de una pieza tiene
$\sigma_y = 100$, $\sigma_z = -40$, $\tau_{yz} = 50$ MPa.
El material es un acero con límite elástico
$\sigma_{e} = 320$ MPa.

1. Calcula las tensiones principales $\sigma_1$, $\sigma_2$.
2. Calcula la cortante máxima en el plano y compárala con la cortante
   máxima 3D (recuerda que $\sigma_x = 0$ es la tercera principal en
   estado plano).
3. Aplica el criterio de Tresca,
   $n = \sigma_{e}/(2\tau_{\max,\,3D})$, y comenta si la pieza tiene
   coeficiente de seguridad aceptable ($n \ge 2$).
""",
    r"""
**1) Tensiones principales en el plano YZ:**

$$C = (100 - 40)/2 = 30,\qquad R = \sqrt{((100-(-40))/2)^2 + 50^2} = \sqrt{70^2 + 50^2}$$

$$R = \sqrt{4900 + 2500} = \sqrt{7400} \approx 86{,}02\ \text{MPa}$$

$$\sigma_1 = 30 + 86{,}02 \approx 116{,}02\ \text{MPa},\qquad \sigma_2 = 30 - 86{,}02 \approx -56{,}02\ \text{MPa}$$

**2) Cortantes máximas:**

- En el plano: $\tau_{\max,\,2D} = R \approx 86{,}02$ MPa.
- En 3D, las tres principales son
  $\sigma_1 \approx 116{,}0$, $\sigma_x = 0$, $\sigma_2 \approx -56{,}0$,
  ya ordenadas. La cortante máxima 3D es

$$\tau_{\max,\,3D} = \tfrac{\sigma_1 - \sigma_2}{2} \approx 86{,}02\ \text{MPa}$$

Como $\sigma_x = 0$ queda **entre** las dos principales del plano, la
cortante 3D coincide con la 2D.

**3) Tresca:**

$$n = \frac{\sigma_e}{2\,\tau_{\max,\,3D}} = \frac{320}{2\cdot 86{,}02} \approx 1{,}86$$

Está **por debajo del coeficiente de seguridad recomendado**
$n \ge 2$, por lo que la pieza está al límite y conviene reducir las
tensiones (más sección, otro material o redistribuir esfuerzos).
""",
)


_ejercicio(
    "Ejercicio 6 — Cortante máxima y plano principal",
    r"""
Demuestra geométricamente, usando el círculo de Mohr, que en cualquier
estado plano se cumple:

$$\tau_{\max} = \frac{\sigma_1 - \sigma_2}{2}$$

y que el plano donde aparece $\tau_{\max}$ está **a $45^{\circ}$** del
plano principal. ¿Cuál es la tensión normal sobre ese plano?
""",
    r"""
En el círculo, los puntos $\sigma_1$ y $\sigma_2$ están sobre el eje
horizontal $\tau = 0$, separados por la distancia $2R$.

El punto más alto del círculo está en
$(C, R)$, con $C = (\sigma_1+\sigma_2)/2$ y radio
$R = (\sigma_1-\sigma_2)/2$. Por tanto, **la cortante máxima vale el
radio**:

$$\tau_{\max} = R = \frac{\sigma_1 - \sigma_2}{2}.$$

El ángulo recorrido sobre el círculo desde $\sigma_1$ hasta el punto
$(C, R)$ es $90^{\circ}$. Como el círculo de Mohr "duplica el ángulo
físico", **en el espacio físico** la rotación necesaria es
$45^{\circ}$.

La **tensión normal sobre el plano de cortante máxima** es la abscisa
del punto $(C, R)$, es decir,

$$\sigma_n = C = \tfrac{\sigma_1 + \sigma_2}{2}.$$

El plano de máxima cortante **no es** un plano libre de tensión
normal: salvo que el estado sea cortante puro ($C = 0$), siempre tiene
una σ residual igual a la media de las dos principales.

> *Verificación en el explorador*: para cualquier estado, selecciona
> "Cortante máxima (+)" y comprueba que el punto naranja cae justo
> sobre $(C, R)$ y a $\alpha = \theta_p + 45^{\circ}$.
""",
)


st.markdown("---")
st.caption(
    "Consejo: usa estos ejercicios como apoyo. Si te equivocas, no "
    "mires la solución inmediatamente: vuelve al explorador, fija ese "
    "estado tensional y arrastra el plano hasta entender qué número "
    "no encaja."
)
