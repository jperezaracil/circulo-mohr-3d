"""
Círculos de Mohr 3D — Aplicación interactiva con Streamlit
==========================================================

Cómo lanzar localmente:
    pip install streamlit numpy matplotlib
    streamlit run app_streamlit.py

Cómo desplegar gratis para los alumnos:
    1. Sube este archivo y un `requirements.txt` a un repositorio público de GitHub
    2. Entra en https://streamlit.io/cloud y conecta tu repo
    3. Comparte la URL con los alumnos (no necesitan instalar nada)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Círculos de Mohr 3D",
                   layout="wide", page_icon="◯")

st.title("Círculos de Mohr 3D — Visualización interactiva")
st.markdown(
    """
    Modifica las **6 componentes del tensor de tensiones** y los **3 ángulos
    de rotación** del cubo en la barra lateral. El panel izquierdo muestra el
    cubo girado con las tensiones que actúan sobre cada cara, y el derecho los
    tres círculos de Mohr con la **región admisible** en amarillo.

    El caso **2D** se obtiene poniendo $\\sigma_z = \\tau_{xz} = \\tau_{yz} = 0$
    y rotando solo $\\gamma$.
    """
)

# ---------------------------------------------------------------------------
# Funciones del modelo
# ---------------------------------------------------------------------------
def matriz_tension(sx, sy, sz, txy, txz, tyz):
    return np.array([[sx,  txy, txz],
                     [txy, sy,  tyz],
                     [txz, tyz, sz ]], dtype=float)

def matriz_rotacion(alpha_deg, beta_deg, gamma_deg):
    a, b, g = np.deg2rad([alpha_deg, beta_deg, gamma_deg])
    Rx = np.array([[1, 0,         0        ],
                   [0, np.cos(a), -np.sin(a)],
                   [0, np.sin(a),  np.cos(a)]])
    Ry = np.array([[ np.cos(b), 0, np.sin(b)],
                   [ 0,         1, 0        ],
                   [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(g), -np.sin(g), 0],
                   [np.sin(g),  np.cos(g), 0],
                   [0,          0,         1]])
    return Rz @ Ry @ Rx

def transformar_tensor(T, R):
    return R.T @ T @ R

def tensiones_principales_3d(T):
    vals, vecs = np.linalg.eigh(T)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx]

def estado_caras(Tp):
    out = []
    for i in range(3):
        t = Tp[:, i]
        sn = Tp[i, i]
        tn = np.sqrt(max(np.dot(t, t) - sn**2, 0.0))
        out.append((sn, tn))
    return out

# ---------------------------------------------------------------------------
# Funciones de dibujo
# ---------------------------------------------------------------------------
def dibujar_cubo_3d(ax, T0, R):
    ax.clear()
    L = 1.0
    Tp = transformar_tensor(T0, R)
    corners0 = np.array([[a, b, c]
                         for a in (-L, L) for b in (-L, L) for c in (-L, L)])
    edges_idx = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),
                 (2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for i, j in edges_idx:
        ax.plot(*zip(corners0[i], corners0[j]),
                color='gray', lw=0.6, ls=':', alpha=0.4)
    corners = corners0 @ R.T
    for i, j in edges_idx:
        ax.plot(*zip(corners[i], corners[j]), color='black', lw=1.6)

    smax = max(np.abs(T0).max(), 1e-9)
    k = 0.7 / smax
    colores_caras = ['#d62728', '#2ca02c', '#1f77b4']
    nombres_caras = ["x'", "y'", "z'"]
    for i in range(3):
        n  = R[:, i]
        ej = R[:, (i+1) % 3]
        ek = R[:, (i+2) % 3]
        center_pos =  L * n
        center_neg = -L * n
        sigma_n = Tp[i, i]
        shear_vec = Tp[(i+1) % 3, i] * ej + Tp[(i+2) % 3, i] * ek
        ax.quiver(*center_pos, *(sigma_n*k*n),
                  color='red', lw=2.0, arrow_length_ratio=0.18, normalize=False)
        ax.quiver(*center_neg, *(-sigma_n*k*n),
                  color='red', lw=2.0, arrow_length_ratio=0.18, normalize=False)
        if np.linalg.norm(shear_vec) > 1e-6 * smax:
            ax.quiver(*center_pos, *(shear_vec*k),
                      color='blue', lw=2.0, arrow_length_ratio=0.18, normalize=False)
            ax.quiver(*center_neg, *(-shear_vec*k),
                      color='blue', lw=2.0, arrow_length_ratio=0.18, normalize=False)
        ax.text(*(1.55*n), nombres_caras[i], color=colores_caras[i],
                fontsize=11, fontweight='bold')

    for i, label in enumerate(['x', 'y', 'z']):
        e = np.zeros(3); e[i] = 2.3
        ax.quiver(0, 0, 0, *e, color='gray', lw=1, arrow_length_ratio=0.06)
        ax.text(*(e*1.08), label, color='gray', fontsize=10)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_zlim(-2.5, 2.5)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('Cubo girado y tensiones en sus caras')


def dibujar_mohr_3d(ax, T0, R):
    ax.clear()
    Tp = transformar_tensor(T0, R)
    s_arr, _ = tensiones_principales_3d(T0)
    s1, s2, s3 = s_arr
    C12, R12 = (s1+s2)/2, (s1-s2)/2
    C23, R23 = (s2+s3)/2, (s2-s3)/2
    C13, R13 = (s1+s3)/2, (s1-s3)/2

    if R13 > 1e-9:
        margen = 0.15 * R13 + 0.5
        xs = np.linspace(s3 - margen, s1 + margen, 500)
        ys = np.linspace(0,             R13 + margen, 300)
        X, Y = np.meshgrid(xs, ys)
        region = (((X - C13)**2 + Y**2 <= R13**2) &
                  ((X - C12)**2 + Y**2 >= R12**2) &
                  ((X - C23)**2 + Y**2 >= R23**2))
        ax.contourf(X, Y, region.astype(float),
                    levels=[0.5, 1.5], colors=['#fff3b0'], alpha=0.55)

    th = np.linspace(0, np.pi, 200)
    ax.plot(C12 + R12*np.cos(th), R12*np.sin(th),
            color='#1f77b4', lw=1.6, label=f'C₁₂  R={R12:.1f}')
    ax.plot(C23 + R23*np.cos(th), R23*np.sin(th),
            color='#2ca02c', lw=1.6, label=f'C₂₃  R={R23:.1f}')
    ax.plot(C13 + R13*np.cos(th), R13*np.sin(th),
            color='#d62728', lw=1.8, label=f'C₁₃  R={R13:.1f} (máx)')

    for sigma, lab in zip([s1, s2, s3], [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']):
        ax.plot(sigma, 0, 'ko', markersize=6, zorder=4)
        ax.annotate(f'{lab}={sigma:.1f}', xy=(sigma, 0),
                    xytext=(0, -14), textcoords='offset points',
                    ha='center', va='top', fontsize=9)

    estado = estado_caras(Tp)
    colores = ['#d62728', '#2ca02c', '#1f77b4']
    nombres = ["x'", "y'", "z'"]
    for i, (sn, tn) in enumerate(estado):
        ax.plot(sn, tn, 'o', color=colores[i], markersize=11, zorder=6,
                markeredgecolor='black', markeredgewidth=0.6)
        ax.annotate(f"  {nombres[i]} ({sn:.1f}, {tn:.1f})",
                    xy=(sn, tn), color=colores[i], fontsize=9, va='center')

    ax.axhline(0, color='gray', lw=0.7)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$|\tau|$')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(rf'Círculos de Mohr 3D · $\tau_{{máx}}={R13:.2f}$')

# ---------------------------------------------------------------------------
# Sidebar — controles
# ---------------------------------------------------------------------------
st.sidebar.header("Tensor de tensiones inicial")
sigma_x = st.sidebar.slider("σx",  -100, 100, 80, 5)
sigma_y = st.sidebar.slider("σy",  -100, 100, 20, 5)
sigma_z = st.sidebar.slider("σz",  -100, 100,  0, 5)
tau_xy  = st.sidebar.slider("τxy", -80,  80,  30, 5)
tau_xz  = st.sidebar.slider("τxz", -80,  80,   0, 5)
tau_yz  = st.sidebar.slider("τyz", -80,  80,   0, 5)

st.sidebar.header("Rotación del cubo")
alpha = st.sidebar.slider("α — giro alrededor de x (°)", -90, 90, 0, 1)
beta  = st.sidebar.slider("β — giro alrededor de y (°)", -90, 90, 0, 1)
gamma = st.sidebar.slider("γ — giro alrededor de z (°)", -90, 90, 0, 1)

st.sidebar.markdown("---")
preset = st.sidebar.selectbox(
    "Casos didácticos (preset)",
    ["—", "Tracción uniaxial", "Cortante puro 2D",
     "Estado hidrostático", "Cortante 3D genérico"]
)
st.sidebar.caption(
    "Selecciona un preset para fijar el tensor; sigue moviendo los ángulos "
    "para ver cómo se mueven los puntos sobre los círculos."
)

if preset == "Tracción uniaxial":
    sigma_x, sigma_y, sigma_z = 100, 0, 0
    tau_xy = tau_xz = tau_yz = 0
elif preset == "Cortante puro 2D":
    sigma_x = sigma_y = sigma_z = 0
    tau_xy, tau_xz, tau_yz = 50, 0, 0
elif preset == "Estado hidrostático":
    sigma_x = sigma_y = sigma_z = 50
    tau_xy = tau_xz = tau_yz = 0
elif preset == "Cortante 3D genérico":
    sigma_x, sigma_y, sigma_z = 80, 20, -30
    tau_xy, tau_xz, tau_yz = 25, 15, 10

# ---------------------------------------------------------------------------
# Cálculos y figura
# ---------------------------------------------------------------------------
T0 = matriz_tension(sigma_x, sigma_y, sigma_z, tau_xy, tau_xz, tau_yz)
R  = matriz_rotacion(alpha, beta, gamma)
Tp = transformar_tensor(T0, R)
s_arr, _ = tensiones_principales_3d(T0)

fig = plt.figure(figsize=(14, 6.5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)
dibujar_cubo_3d(ax1, T0, R)
dibujar_mohr_3d(ax2, T0, R)
plt.tight_layout()
st.pyplot(fig)

# ---------------------------------------------------------------------------
# Tabla de resultados
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Tensor inicial σ")
    st.latex(rf"""
\begin{{pmatrix}}
{sigma_x:+.1f} & {tau_xy:+.1f} & {tau_xz:+.1f} \\
{tau_xy:+.1f} & {sigma_y:+.1f} & {tau_yz:+.1f} \\
{tau_xz:+.1f} & {tau_yz:+.1f} & {sigma_z:+.1f}
\end{{pmatrix}}
""")

with col2:
    st.subheader("Tensor rotado σ′")
    st.latex(rf"""
\begin{{pmatrix}}
{Tp[0,0]:+.1f} & {Tp[0,1]:+.1f} & {Tp[0,2]:+.1f} \\
{Tp[0,1]:+.1f} & {Tp[1,1]:+.1f} & {Tp[1,2]:+.1f} \\
{Tp[0,2]:+.1f} & {Tp[1,2]:+.1f} & {Tp[2,2]:+.1f}
\end{{pmatrix}}
""")

with col3:
    st.subheader("Tensiones principales")
    st.latex(rf"\sigma_1 = {s_arr[0]:+.2f}")
    st.latex(rf"\sigma_2 = {s_arr[1]:+.2f}")
    st.latex(rf"\sigma_3 = {s_arr[2]:+.2f}")
    st.latex(rf"\tau_{{máx}} = {(s_arr[0]-s_arr[2])/2:.2f}")
