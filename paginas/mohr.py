"""
Círculos de Mohr 3D — Aplicación interactiva con Streamlit
==========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                     # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import streamlit as st


st.title("Círculos de Mohr 3D — Visualización interactiva")
st.markdown(
    """
    Modifica las **6 componentes del tensor de tensiones** y los **3 ángulos
    de rotación** del cubo en la barra lateral. El panel izquierdo muestra el
    cubo girado con las tensiones que actúan sobre cada cara, y el derecho los
    tres círculos de Mohr con la **región admisible** en amarillo.

    Activa abajo la opción **"Plano oblicuo"** para calcular las tensiones que
    actúan sobre un plano arbitrario definido por su normal — incluyendo el
    vector tracción y la descomposición en ejes locales del plano.

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
    """Para cada cara del cubo girado: (σ_n, |τ_n|)."""
    out = []
    for i in range(3):
        t = Tp[:, i]
        sn = Tp[i, i]
        tn = np.sqrt(max(np.dot(t, t) - sn**2, 0.0))
        out.append((sn, tn))
    return out


def localizar_punto_3d(s_arr, vecs, sigma_q, tau_q):
    """
    Localiza un punto (σ_q, |τ_q|) sobre el diagrama de Mohr 3D.

    s_arr   [σ1, σ2, σ3] tensiones principales (σ1 ≥ σ2 ≥ σ3).
    vecs    matriz 3×3 con las direcciones principales por columnas
            (en coords globales x, y, z).
    sigma_q tensión normal consultada (escalar).
    tau_q   |τ| consultado (≥ 0).

    Devuelve dict con admisibilidad, cosenos² del vector normal en ejes
    principales, identificación del círculo sobre el que cae (si lo hace),
    y un vector normal candidato (en coords globales).
    """
    s1, s2, s3 = (float(s_arr[0]), float(s_arr[1]), float(s_arr[2]))
    s = float(sigma_q)
    tau = float(abs(tau_q))

    C12, R12 = (s1 + s2) / 2.0, (s1 - s2) / 2.0
    C23, R23 = (s2 + s3) / 2.0, (s2 - s3) / 2.0
    C13, R13 = (s1 + s3) / 2.0, (s1 - s3) / 2.0

    EPS = 1e-9
    den_l = (s1 - s2) * (s1 - s3)
    den_m = (s2 - s3) * (s2 - s1)
    den_n = (s3 - s1) * (s3 - s2)
    l2 = (((s - s2) * (s - s3) + tau**2) / den_l) if abs(den_l) > EPS else float('nan')
    m2 = (((s - s3) * (s - s1) + tau**2) / den_m) if abs(den_m) > EPS else float('nan')
    n2 = (((s - s1) * (s - s2) + tau**2) / den_n) if abs(den_n) > EPS else float('nan')

    scale2 = max(R13**2, 1.0)
    tol = 1e-3 * scale2

    def en_circulo(C, R):
        return abs((s - C)**2 + tau**2 - R**2) <= tol

    en_C13 = en_circulo(C13, R13)
    en_C12 = en_circulo(C12, R12)
    en_C23 = en_circulo(C23, R23)
    nombre_circ = ("C13" if en_C13 else
                   "C12" if en_C12 else
                   "C23" if en_C23 else None)

    dentro_C13 = (s - C13)**2 + tau**2 <= R13**2 + tol
    fuera_C12  = (s - C12)**2 + tau**2 >= R12**2 - tol
    fuera_C23  = (s - C23)**2 + tau**2 >= R23**2 - tol
    admisible  = bool(dentro_C13 and fuera_C12 and fuera_C23 and tau >= -1e-9)

    if admisible and all(x == x for x in (l2, m2, n2)):
        l = float(np.sqrt(max(l2, 0.0)))
        m = float(np.sqrt(max(m2, 0.0)))
        n_ = float(np.sqrt(max(n2, 0.0)))
        # Vector normal en coords globales (eligiendo signos +,+,+)
        n_principal = np.array([l, m, n_])
        n_global = vecs @ n_principal
        nrm = np.linalg.norm(n_global)
        if nrm > 1e-12:
            n_global = n_global / nrm
    else:
        l = m = n_ = float('nan')
        n_global = None

    return {
        "sigma_q": s, "tau_q": tau,
        "admisible": admisible,
        "l2": l2, "m2": m2, "n2": n2,
        "l": l, "m": m, "n": n_,
        "n_global": n_global,
        "en_circulo": nombre_circ,
        "C12": C12, "R12": R12,
        "C23": C23, "R23": R23,
        "C13": C13, "R13": R13,
        "dentro_C13": bool(dentro_C13),
        "fuera_C12":  bool(fuera_C12),
        "fuera_C23":  bool(fuera_C23),
    }


def tensiones_en_plano(T0, theta_deg, phi_deg):
    """
    Tensiones sobre un plano arbitrario, definido por la dirección de su
    normal en coordenadas esféricas (θ desde el eje z, φ azimut desde el
    eje x).

    Los **ejes locales del plano** son la terna ortonormal (n, eθ, eφ),
    donde n es la normal y (eθ, eφ) son dos direcciones tangentes en el
    plano (las direcciones naturales de aumento de θ y φ respectivamente).

    Devuelve un dict con:
        n, e_theta, e_phi    Vectores unitarios en coords globales
        t                    Vector tracción t = σ·n   (coords globales)
        sigma_n              Tensión normal al plano (escalar)
        tau_vec              Vector cortante τ = t − σ_n n (coords globales)
        tau_mag              |τ| (magnitud de la cortante en el plano)
        tau_theta, tau_phi   Componentes de τ en los ejes locales del plano
    """
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)

    n = np.array([np.sin(th) * np.cos(ph),
                  np.sin(th) * np.sin(ph),
                  np.cos(th)])
    e_theta = np.array([ np.cos(th) * np.cos(ph),
                         np.cos(th) * np.sin(ph),
                        -np.sin(th)])
    e_phi   = np.array([-np.sin(ph),
                         np.cos(ph),
                         0.0])

    t = T0 @ n
    sigma_n = float(np.dot(t, n))
    tau_vec = t - sigma_n * n
    tau_mag = float(np.sqrt(max(np.dot(t, t) - sigma_n**2, 0.0)))
    tau_theta = float(np.dot(tau_vec, e_theta))
    tau_phi   = float(np.dot(tau_vec, e_phi))

    return {
        "n": n, "e_theta": e_theta, "e_phi": e_phi,
        "t": t,
        "sigma_n": sigma_n,
        "tau_vec": tau_vec,
        "tau_mag": tau_mag,
        "tau_theta": tau_theta, "tau_phi": tau_phi,
    }


# ---------------------------------------------------------------------------
# Funciones de dibujo
# ---------------------------------------------------------------------------
def dibujar_cubo_3d(ax, T0, R, plano=None):
    ax.clear()
    L = 1.0
    Tp = transformar_tensor(T0, R)

    corners0 = np.array([[a, b, c]
                         for a in (-L, L) for b in (-L, L) for c in (-L, L)])
    edges_idx = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),
                 (2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

    # Cubo de referencia (sin girar)
    for i, j in edges_idx:
        ax.plot(*zip(corners0[i], corners0[j]),
                color='gray', lw=0.6, ls=':', alpha=0.4)

    # Cubo girado
    corners = corners0 @ R.T
    for i, j in edges_idx:
        ax.plot(*zip(corners[i], corners[j]), color='black', lw=1.6)

    # Tensiones en cada cara
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

    # Plano oblicuo (si está activo)
    if plano is not None:
        n_plano = plano["n"]
        e1 = plano["e_theta"]
        e2 = plano["e_phi"]

        # 1) Patch del plano (atraviesa el cubo, centrado en origen)
        size = 1.6 * L
        verts = np.array([
             size*e1 + size*e2,
            -size*e1 + size*e2,
            -size*e1 - size*e2,
             size*e1 - size*e2,
        ])
        poly = Poly3DCollection([verts], alpha=0.22,
                                facecolor='magenta', edgecolor='magenta', lw=1.2)
        ax.add_collection3d(poly)

        # 2) Normal del plano (verde, fina) — referencia de orientación
        ax.quiver(0, 0, 0, *(1.9*n_plano),
                  color='darkgreen', lw=1.4,
                  arrow_length_ratio=0.10, normalize=False)
        ax.text(*(2.05*n_plano), r"$\vec{n}$",
                color='darkgreen', fontsize=11, fontweight='bold')

        # 3) Vector tracción t (morado, grueso)
        t_vec = plano["t"]
        ax.quiver(0, 0, 0, *(t_vec*k),
                  color='purple', lw=3.0,
                  arrow_length_ratio=0.15, normalize=False)
        ax.text(*(t_vec*k*1.1), r"$\vec{t}$",
                color='purple', fontsize=11, fontweight='bold')

        # 4) Descomposición de t (en discontinuo): σ_n·n + τ_vec
        sigma_n = plano["sigma_n"]
        tau_vec = plano["tau_vec"]
        ax.quiver(0, 0, 0, *(sigma_n*k*n_plano),
                  color='#c0392b', lw=1.5, ls='--',
                  arrow_length_ratio=0.12, normalize=False)
        if np.linalg.norm(tau_vec) > 1e-6 * smax:
            ax.quiver(0, 0, 0, *(tau_vec*k),
                      color='#1a5276', lw=1.5, ls='--',
                      arrow_length_ratio=0.12, normalize=False)

    # Ejes globales
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
    titulo = 'Cubo girado y tensiones en sus caras'
    if plano is not None:
        titulo += '\n+ plano oblicuo (magenta) y tracción $\\vec{t}$ (morado)'
    ax.set_title(titulo)


def dibujar_mohr_3d(ax, T0, R, plano=None, consulta=None):
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

    for sigma, lab in zip([s1, s2, s3],
                          [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']):
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

    # Punto del plano oblicuo (estrella magenta)
    if plano is not None:
        sn_pl = plano["sigma_n"]
        tn_pl = plano["tau_mag"]
        ax.plot(sn_pl, tn_pl, marker='*', color='magenta', markersize=20,
                markeredgecolor='black', markeredgewidth=0.8, zorder=7,
                label='Plano oblicuo')
        ax.annotate(f"  Plano ({sn_pl:.1f}, {tn_pl:.1f})",
                    xy=(sn_pl, tn_pl), color='magenta',
                    fontsize=10, fontweight='bold', va='center')

    # Punto de consulta (σ_q, |τ_q|)
    if consulta is not None:
        sq = consulta["sigma_q"]
        tq = consulta["tau_q"]
        col = '#e67e22' if consulta["admisible"] else '#7f8c8d'
        etiqueta = ('Consulta (admisible)' if consulta["admisible"]
                    else 'Consulta (NO admisible)')
        ax.plot(sq, tq, marker='D', color=col, markersize=13,
                markeredgecolor='black', markeredgewidth=0.8, zorder=8,
                label=etiqueta)
        ax.annotate(f"  ({sq:.1f}, {tq:.1f})",
                    xy=(sq, tq), color=col,
                    fontsize=10, fontweight='bold', va='center')

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
    ["—", "Estado plano (XY)", "Tracción uniaxial", "Cortante puro 2D",
     "Torsión pura (barra eje X)", "Estado hidrostático",
     "Cortante 3D genérico"]
)
st.sidebar.caption(
    "Selecciona un preset para fijar el tensor; sigue moviendo los ángulos "
    "para ver cómo se mueven los puntos sobre los círculos."
)

if preset == "Estado plano (XY)":
    # Caso 2D clásico: solo σx, σy, τxy en el plano xy.
    # Todo lo que sale del plano vale 0, así que el cubo solo siente
    # tensiones en sus caras x e y. La cara z queda libre (σz = 0).
    sigma_x, sigma_y, sigma_z = 80, 20, 0
    tau_xy, tau_xz, tau_yz    = 30, 0, 0
elif preset == "Tracción uniaxial":
    sigma_x, sigma_y, sigma_z = 100, 0, 0
    tau_xy = tau_xz = tau_yz = 0
elif preset == "Cortante puro 2D":
    sigma_x = sigma_y = sigma_z = 0
    tau_xy, tau_xz, tau_yz = 50, 0, 0
elif preset == "Torsión pura (barra eje X)":
    # Estado tensional en un punto de la superficie de un eje cilíndrico
    # sometido a torsión pura. La tensión cortante τxy = T·r/J vale 60 MPa.
    # X = eje longitudinal, Y = tangente a la superficie, Z = radial.
    # Las tensiones normales son nulas. Las direcciones principales son
    # las hélices a ±45° respecto al eje X.
    sigma_x = sigma_y = sigma_z = 0
    tau_xy, tau_xz, tau_yz = 60, 0, 0
elif preset == "Estado hidrostático":
    sigma_x = sigma_y = sigma_z = 50
    tau_xy = tau_xz = tau_yz = 0
elif preset == "Cortante 3D genérico":
    sigma_x, sigma_y, sigma_z = 80, 20, -30
    tau_xy, tau_xz, tau_yz = 25, 15, 10

# --- Plano oblicuo arbitrario ---
st.sidebar.markdown("---")
st.sidebar.header("Plano oblicuo (opcional)")
mostrar_plano = st.sidebar.checkbox(
    "Tensiones sobre un plano de orientación arbitraria",
    value=False,
    help="Define un plano por la dirección de su normal y se calcula el "
         "vector tracción y sus componentes (σ_n y τ).",
)
if mostrar_plano:
    theta_p = st.sidebar.slider("θ — inclinación de n̂ desde eje z (°)",
                                0, 180, 45, 1)
    phi_p   = st.sidebar.slider("φ — azimut de n̂ desde eje x (°)",
                                0, 360, 0, 1)
else:
    theta_p, phi_p = None, None

# --- Consulta de un punto (σ_n, |τ|) ---
st.sidebar.markdown("---")
st.sidebar.header("Consultar punto (σ, |τ|)")
activar_consulta_3d = st.sidebar.checkbox(
    "Introducir (σ_n, |τ|) y localizarlo en el diagrama",
    value=False,
    help="Comprueba si un par (σ_n, |τ|) cae dentro de la región admisible "
         "(zona amarilla) o sobre alguno de los tres círculos. Si es "
         "admisible, calcula los cosenos² del vector normal en ejes "
         "principales y una orientación posible del plano."
)
if activar_consulta_3d:
    sigma_q_3d = st.sidebar.number_input("σ_n consultado", value=40.0,
                                         step=5.0, format="%.2f")
    tau_q_3d   = st.sidebar.number_input("|τ| consultado (≥ 0)",
                                         value=30.0, step=5.0,
                                         min_value=0.0, format="%.2f")
else:
    sigma_q_3d, tau_q_3d = None, None


# ---------------------------------------------------------------------------
# Cálculos y figura
# ---------------------------------------------------------------------------
T0 = matriz_tension(sigma_x, sigma_y, sigma_z, tau_xy, tau_xz, tau_yz)
R  = matriz_rotacion(alpha, beta, gamma)
Tp = transformar_tensor(T0, R)
s_arr, vecs = tensiones_principales_3d(T0)

plano = tensiones_en_plano(T0, theta_p, phi_p) if mostrar_plano else None
consulta_3d = (localizar_punto_3d(s_arr, vecs, sigma_q_3d, tau_q_3d)
               if activar_consulta_3d else None)

fig = plt.figure(figsize=(14, 6.5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)
dibujar_cubo_3d(ax1, T0, R, plano=plano)
dibujar_mohr_3d(ax2, T0, R, plano=plano, consulta=consulta_3d)
plt.tight_layout()
st.pyplot(fig)


# ---------------------------------------------------------------------------
# Explicación: qué representan los puntos del Mohr y por qué se mueven
# ---------------------------------------------------------------------------
with st.expander(
    "¿Qué representan los tres puntos de colores y por qué se mueven al "
    "girar el cubo?"
):
    st.markdown(
        r"""
**Los tres puntos sobre el círculo de Mohr** corresponden a las **tres
caras del cubo girado**:

- **Punto rojo** $x'$ — tensiones sobre la cara cuya normal es el eje
  $x'$ del cubo girado: $(\sigma_{x'},\ |\tau|_{\text{cara }x'})$.
- **Punto verde** $y'$ — igual para la cara $y'$.
- **Punto azul** $z'$ — igual para la cara $z'$.

#### Por qué se mueven al rotar

Las caras del cubo son tres **planos** que pasan por el mismo punto del
sólido. Rotar el cubo significa **elegir otros tres planos
perpendiculares** sobre los que medir las tensiones. Como las
tensiones que actúan sobre un plano dependen de la orientación del
plano, al cambiar los planos cambian los pares $(\sigma_n, |\tau|)$
correspondientes — y por tanto los puntos se mueven.

Es el mismo principio que en el círculo de Mohr 2D: cuando giras el
elemento un ángulo $\theta$, el punto recorre el círculo. En 3D
ocurre lo mismo, solo que hay **tres caras** (tres puntos) moviéndose
simultáneamente, cada uno por su propia trayectoria dentro de la
región admisible.

#### Qué *no* cambia con la rotación

- **Los tres círculos** son fijos. Dependen únicamente de las tensiones
  principales $\sigma_1,\sigma_2,\sigma_3$, que son **invariantes** del
  estado tensional (no dependen del sistema de referencia).
- **Las tensiones principales** marcadas en negro sobre el eje $\sigma$
  no se mueven.
- **La región amarilla** es la misma. Y los tres puntos, por mucho que
  los muevas, **nunca pueden salir de ella** — sería físicamente
  imposible.

#### Casos límite que lo confirman

- **Todos los ángulos a 0**: los tres puntos quedan en
  $(\sigma_x, |\tau|_{\text{cara }x})$, $(\sigma_y, |\tau|_{\text{cara }y})$,
  $(\sigma_z, |\tau|_{\text{cara }z})$ — exactamente las tensiones del
  tensor inicial sobre las caras $x, y, z$.
- **Cubo alineado con las direcciones principales**: los tres puntos
  caen sobre el eje $\sigma$ con $|\tau|=0$. Las tres caras del cubo
  son entonces los planos principales y solo soportan tensión normal
  pura.
- **Estado hidrostático** ($\sigma_x = \sigma_y = \sigma_z$, sin
  cortantes): los círculos degeneran en un punto y los tres puntos del
  cubo se quedan ahí, gires lo que gires. Es la única configuración en
  la que rotar el cubo no afecta a los puntos.
        """
    )


# ---------------------------------------------------------------------------
# Resultados — tensor inicial / rotado / tensiones principales
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


# ---------------------------------------------------------------------------
# Resultados — plano oblicuo
# ---------------------------------------------------------------------------
if plano is not None:
    st.markdown("---")
    st.subheader("Tensiones sobre el plano oblicuo")

    n_pl = plano["n"]
    t_pl = plano["t"]

    cA, cB = st.columns(2)

    with cA:
        st.markdown("**Geometría del plano**")
        st.latex(rf"\vec{{n}} = ({n_pl[0]:+.3f},\ {n_pl[1]:+.3f},\ {n_pl[2]:+.3f})")
        st.caption("(cosenos directores de la normal)")

        st.markdown("**Vector tracción** $\\vec{t} = \\boldsymbol{\\sigma}\\cdot\\vec{n}$")
        st.caption("expresado en ejes globales (x, y, z)")
        st.latex(rf"""
\vec{{t}} =
\begin{{pmatrix}}
{t_pl[0]:+.3f} \\ {t_pl[1]:+.3f} \\ {t_pl[2]:+.3f}
\end{{pmatrix}}
""")

    with cB:
        st.markdown("**Componentes en ejes locales del plano**")
        st.caption(
            "Los ejes locales son $(\\vec{n},\\, \\vec{e}_\\theta,\\, \\vec{e}_\\varphi)$, "
            "donde $\\vec{n}$ es la normal y $(\\vec{e}_\\theta, \\vec{e}_\\varphi)$ son "
            "dos direcciones tangentes en el plano."
        )
        st.latex(rf"\sigma_n = \vec{{t}}\cdot\vec{{n}} = {plano['sigma_n']:+.3f}")
        st.latex(rf"|\vec{{\tau}}| = \sqrt{{|\vec{{t}}|^2 - \sigma_n^2}} = {plano['tau_mag']:.3f}")
        st.latex(rf"\tau_{{e_\theta}} = {plano['tau_theta']:+.3f}")
        st.latex(rf"\tau_{{e_\varphi}} = {plano['tau_phi']:+.3f}")

    st.warning(
        "**Cuidado con los ejes**: los valores $\\sigma_n,\\ \\tau_{e_\\theta},\\ "
        "\\tau_{e_\\varphi}$ están expresados en los **ejes locales del plano**, no "
        "en los ejes globales (x, y, z). $\\sigma_n$ es la proyección de $\\vec{t}$ "
        "sobre la normal, y $\\tau_{e_\\theta},\\ \\tau_{e_\\varphi}$ son las "
        "proyecciones del vector cortante $\\vec{\\tau}$ sobre las dos tangentes "
        "del plano. La magnitud $|\\vec{\\tau}|$ es la que aparece sobre el círculo "
        "de Mohr (estrella magenta) y siempre cae dentro de la región amarilla."
    )

    with st.expander("Definición de los ejes locales"):
        st.markdown(
            r"""
La dirección normal al plano se parametriza con dos ángulos esféricos:

$$\vec{n} = (\sin\theta\cos\varphi,\ \sin\theta\sin\varphi,\ \cos\theta)$$

Las dos direcciones tangentes naturales (las del incremento de $\theta$ y
$\varphi$) son:

$$\vec{e}_\theta = (\cos\theta\cos\varphi,\ \cos\theta\sin\varphi,\ -\sin\theta)$$

$$\vec{e}_\varphi = (-\sin\varphi,\ \cos\varphi,\ 0)$$

La terna $(\vec{n},\ \vec{e}_\theta,\ \vec{e}_\varphi)$ es ortonormal y forma
los **ejes locales del plano**. El vector tracción se descompone como:

$$\vec{t} = \sigma_n\,\vec{n} + \tau_{e_\theta}\,\vec{e}_\theta + \tau_{e_\varphi}\,\vec{e}_\varphi$$

con $\sigma_n = \vec{t}\cdot\vec{n}$, $\tau_{e_\theta} = \vec{\tau}\cdot\vec{e}_\theta$
y $\tau_{e_\varphi} = \vec{\tau}\cdot\vec{e}_\varphi$. La cortante en el plano
es el vector $\vec{\tau} = \vec{t} - \sigma_n\vec{n}$, contenido en el plano,
de magnitud $|\vec{\tau}| = \sqrt{\tau_{e_\theta}^2 + \tau_{e_\varphi}^2}$.
            """
        )


# ---------------------------------------------------------------------------
# Resultados — punto de consulta (σ, |τ|)
# ---------------------------------------------------------------------------
if consulta_3d is not None:
    st.markdown("---")
    st.subheader("Punto de consulta (σ, |τ|) en el diagrama 3D")

    cA, cB = st.columns(2)

    with cA:
        st.markdown("**Punto introducido**")
        st.latex(
            rf"(\sigma_n,\ |\tau|) = "
            rf"({consulta_3d['sigma_q']:+.2f},\ {consulta_3d['tau_q']:.2f})"
        )
        st.markdown("**Círculos del estado**")
        st.latex(
            rf"C_{{13}} = {consulta_3d['C13']:+.2f},\quad "
            rf"R_{{13}} = {consulta_3d['R13']:.2f}"
        )
        st.latex(
            rf"C_{{12}} = {consulta_3d['C12']:+.2f},\quad "
            rf"R_{{12}} = {consulta_3d['R12']:.2f}"
        )
        st.latex(
            rf"C_{{23}} = {consulta_3d['C23']:+.2f},\quad "
            rf"R_{{23}} = {consulta_3d['R23']:.2f}"
        )

    with cB:
        st.markdown("**Resultado**")
        if consulta_3d["admisible"]:
            if consulta_3d["en_circulo"] == "C13":
                st.success(
                    "El punto cae **sobre el círculo $C_{13}$** "
                    "(el grande). El plano que ve este estado contiene a "
                    "$\\sigma_1$ y $\\sigma_3$ y es **perpendicular a la "
                    "dirección principal $\\sigma_2$** "
                    "($m^2 = 0$)."
                )
            elif consulta_3d["en_circulo"] == "C12":
                st.success(
                    "El punto cae **sobre el círculo $C_{12}$**. El plano "
                    "que ve este estado contiene a $\\sigma_1$ y "
                    "$\\sigma_2$ y es **perpendicular a la dirección "
                    "principal $\\sigma_3$** ($n^2 = 0$)."
                )
            elif consulta_3d["en_circulo"] == "C23":
                st.success(
                    "El punto cae **sobre el círculo $C_{23}$**. El plano "
                    "que ve este estado contiene a $\\sigma_2$ y "
                    "$\\sigma_3$ y es **perpendicular a la dirección "
                    "principal $\\sigma_1$** ($l^2 = 0$)."
                )
            else:
                st.success(
                    "El punto cae **dentro de la región admisible** "
                    "(zona amarilla) pero no sobre ningún círculo: "
                    "corresponde a planos cuya normal no es perpendicular "
                    "a ninguna dirección principal."
                )

            st.markdown(
                "**Cosenos² de la normal en ejes principales** "
                "(fórmulas de Mohr 3D)"
            )
            l2, m2, n2 = consulta_3d["l2"], consulta_3d["m2"], consulta_3d["n2"]
            st.latex(rf"l^2 = \cos^2(\hat n,\, \vec e_1) = {l2:.4f}")
            st.latex(rf"m^2 = \cos^2(\hat n,\, \vec e_2) = {m2:.4f}")
            st.latex(rf"n^2 = \cos^2(\hat n,\, \vec e_3) = {n2:.4f}")
            st.caption(
                f"Suma de cosenos² = {l2 + m2 + n2:.4f} "
                "(debe valer 1; las pequeñas desviaciones son por redondeo)."
            )

            if consulta_3d["n_global"] is not None:
                ng = consulta_3d["n_global"]
                st.markdown(
                    "**Una orientación posible del plano** (vector normal en "
                    "coordenadas globales x, y, z)"
                )
                st.latex(
                    rf"\hat n = ({ng[0]:+.3f},\ {ng[1]:+.3f},\ {ng[2]:+.3f})"
                )
                st.caption(
                    "Hay hasta **4 planos distintos** que ven el mismo "
                    "$(\\sigma_n, |\\tau|)$, correspondientes a las "
                    "combinaciones de signos $(\\pm l,\\, \\pm m,\\, \\pm n)$. "
                    "Aquí se muestra una de ellas (la de signos +,+,+ en ejes "
                    "principales)."
                )
        else:
            st.error(
                "El punto **NO es admisible**: queda fuera de la región "
                "amarilla. No corresponde a ningún plano físico de este "
                "estado tensional."
            )
            razones = []
            if not consulta_3d["dentro_C13"]:
                razones.append(
                    "está **fuera del círculo grande $C_{13}$** "
                    "($|\\tau|$ supera la cortante máxima del estado)"
                )
            if not consulta_3d["fuera_C12"]:
                razones.append(
                    "está **dentro del círculo $C_{12}$** "
                    "(la región amarilla excluye su interior)"
                )
            if not consulta_3d["fuera_C23"]:
                razones.append(
                    "está **dentro del círculo $C_{23}$** "
                    "(la región amarilla excluye su interior)"
                )
            for r in razones:
                st.markdown(f"- {r}")

    st.info(
        "**Interpretación**: el diagrama de Mohr 3D usa siempre $|\\tau| "
        "\\ge 0$ en el eje vertical. Cualquier plano que pase por el "
        "punto del sólido tiene una pareja $(\\sigma_n,\\, |\\tau|)$ "
        "dentro de la región amarilla. Recíprocamente, todo punto de la "
        "región amarilla es realizable por algún plano (de hecho hasta "
        "por 4 planos distintos: combinaciones de signos de los cosenos "
        "directores)."
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


with st.expander("Problema 1 — Estado plano clásico", expanded=False):
    st.markdown(
        r"""
Una pieza está sometida a un estado de tensión plano:

$$\sigma_x = 80\ \text{MPa},\quad \sigma_y = 20\ \text{MPa},\quad \tau_{xy} = 30\ \text{MPa},\quad \sigma_z = \tau_{xz} = \tau_{yz} = 0.$$

Calcula:

1. Las **tensiones principales** $\sigma_1, \sigma_2, \sigma_3$ (en 3D).
2. El **ángulo principal** $\theta_p$ que orienta $\sigma_1$ en el plano $xy$.
3. La **cortante máxima** $\tau_{\max}$ en 3D y compárala con la
   $\tau_{\max}$ que se obtendría considerando solo el plano $xy$.

> **Pista**: en estado plano $\sigma_z = 0$ es siempre una de las tres
> tensiones principales en 3D. La $\tau_{\max}$ 3D no coincide con la 2D.
"""
    )

with st.expander("Solución del Problema 1", expanded=False):
    st.markdown(
        r"""
**1) Tensiones principales 2D:**

$$\sigma_{1,2} = \tfrac{\sigma_x+\sigma_y}{2} \pm \sqrt{\bigl(\tfrac{\sigma_x-\sigma_y}{2}\bigr)^2 + \tau_{xy}^2} = 50 \pm \sqrt{900 + 900} = 50 \pm 42{,}43$$

Como $\sigma_z = 0$ es también principal, ordenando de mayor a menor:

$$\sigma_1 = 92{,}43\ \text{MPa},\quad \sigma_2 = 7{,}57\ \text{MPa},\quad \sigma_3 = 0\ \text{MPa}$$

**2) Ángulo principal:**

$$\tan(2\theta_p) = \frac{2\tau_{xy}}{\sigma_x - \sigma_y} = \frac{60}{60} = 1 \;\Rightarrow\; 2\theta_p = 45^{\circ} \;\Rightarrow\; \theta_p = 22{,}5^{\circ}$$

**3) Cortante máxima:**

- En 2D (solo plano $xy$): $\tau_{\max,xy} = \tfrac{\sigma_1-\sigma_2}{2} = 42{,}43\ \text{MPa}$.
- En 3D (todos los planos posibles): $\tau_{\max} = \tfrac{\sigma_1-\sigma_3}{2} = 46{,}21\ \text{MPa}$.

La cortante máxima 3D es **mayor** porque incluye planos que salen del plano
$xy$. El plano de $\tau_{\max}$ 3D biseca a $\sigma_1$ y $\sigma_3$.

**Verificación con la app**: preset *"Estado plano (XY)"*. El círculo grande
$C_{13}$ pasa por el origen (porque $\sigma_3 = 0$). Su radio es $46{,}21$.
"""
    )


with st.expander("Problema 2 — Tracción uniaxial: tensiones en un plano oblicuo",
                 expanded=False):
    st.markdown(
        r"""
Una barra está sometida a tracción uniaxial pura: $\sigma_x = 120\ \text{MPa}$,
todas las demás componentes nulas. Considera un plano cuyo vector normal
forma un ángulo de $60^{\circ}$ con el eje $x$, contenido en el plano $xy$.

1. Calcula $\sigma_n$ y $|\vec\tau|$ sobre ese plano.
2. ¿Para qué orientación del plano se hace **máxima** la cortante $|\vec\tau|$?
   ¿Cuánto vale ese máximo?
3. Comprueba con la app activando *"Plano oblicuo"* con $\theta = 90^{\circ}$
   y $\varphi = 60^{\circ}$.
"""
    )

with st.expander("Solución del Problema 2", expanded=False):
    st.markdown(
        r"""
Sobre un plano cuya normal forma un ángulo $\alpha$ con el eje de la
tracción uniaxial, las tensiones son (resultado clásico):

$$\sigma_n = \sigma_x \cos^2\alpha,\qquad |\vec\tau| = \sigma_x \sin\alpha\cos\alpha = \tfrac{\sigma_x}{2}\sin(2\alpha).$$

**1) Para $\alpha = 60^{\circ}$:**

$$\sigma_n = 120 \cdot \cos^2 60^{\circ} = 120 \cdot 0{,}25 = 30\ \text{MPa}$$

$$|\vec\tau| = 120 \cdot \sin 60^{\circ} \cdot \cos 60^{\circ} = 120 \cdot 0{,}866 \cdot 0{,}5 = 51{,}96\ \text{MPa}$$

**2) Cortante máxima:**

$|\vec\tau|$ es máximo cuando $\sin(2\alpha) = 1$, esto es $\alpha = 45^{\circ}$:

$$|\vec\tau|_{\max} = \tfrac{\sigma_x}{2} = 60\ \text{MPa}$$

Es el extremo superior del círculo de Mohr (radio $= \sigma_x/2$).

**3) En la app**: con preset *"Tracción uniaxial"* y plano oblicuo
$\theta = 90^{\circ}$, $\varphi = 60^{\circ}$, la estrella magenta cae en
$(30,\ 51{,}96)$, sobre el círculo grande.
"""
    )


with st.expander("Problema 3 — Estado 3D ya principal", expanded=False):
    st.markdown(
        r"""
En cierto punto de un sólido, el sistema $xyz$ coincide con las direcciones
principales del estado tensional:

$$\sigma_x = 100\ \text{MPa},\quad \sigma_y = 40\ \text{MPa},\quad \sigma_z = -20\ \text{MPa},\quad \tau_{xy} = \tau_{xz} = \tau_{yz} = 0.$$

1. ¿Cómo se ven los tres puntos del cubo en el círculo de Mohr cuando los
   ángulos de rotación son $\alpha = \beta = \gamma = 0$? ¿Por qué?
2. Calcula la cortante máxima 3D y la tensión normal sobre el plano donde
   se produce.
3. Comprueba con la app activando un plano oblicuo a $\theta = 45^{\circ},\ \varphi = 0^{\circ}$.
"""
    )

with st.expander("Solución del Problema 3", expanded=False):
    st.markdown(
        r"""
**1)** Como los ejes $xyz$ son ya las direcciones principales y no hay
cortantes, las tensiones que actúan sobre las caras del cubo son
**solo normales**. En el círculo de Mohr, los tres puntos caen sobre el
eje $\sigma$ con $|\tau| = 0$:

$$x' \to (100, 0),\quad y' \to (40, 0),\quad z' \to (-20, 0).$$

**2)** Tensiones principales ya ordenadas:
$\sigma_1 = 100,\ \sigma_2 = 40,\ \sigma_3 = -20$ MPa. Cortante máxima:

$$\tau_{\max} = \tfrac{\sigma_1 - \sigma_3}{2} = \tfrac{100 - (-20)}{2} = 60\ \text{MPa}$$

Tensión normal sobre el plano de $\tau_{\max}$ (que biseca $\sigma_1$ y $\sigma_3$):

$$\sigma_n = \tfrac{\sigma_1 + \sigma_3}{2} = 40\ \text{MPa}$$

**3)** En la app, con plano oblicuo $\theta = 45^{\circ},\ \varphi = 0^{\circ}$ (la normal
biseca los ejes $x$ y $z$, manteniéndose perpendicular al $y$), la estrella
magenta cae en $(40,\ 60)$ — exactamente la cima del círculo grande $C_{13}$.
"""
    )


with st.expander("Problema 4 — Estado hidrostático",
                 expanded=False):
    st.markdown(
        r"""
Un punto está sometido a un estado hidrostático:

$$\sigma_x = \sigma_y = \sigma_z = p,\qquad \tau_{xy} = \tau_{xz} = \tau_{yz} = 0.$$

Demuestra que **sobre cualquier plano** que pase por el punto, la tensión
normal vale $\sigma_n = p$ y la cortante $|\vec\tau| = 0$. Es decir, todas
las direcciones son principales.

1. Demostración analítica.
2. Comprobación con la app: preset *"Estado hidrostático"*. Mueve los
   ángulos $\alpha, \beta, \gamma$. Activa también un plano oblicuo
   arbitrario.
"""
    )

with st.expander("Solución del Problema 4", expanded=False):
    st.markdown(
        r"""
**1)** El tensor de tensiones es $\boldsymbol\sigma = p\,\mathbf{I}$
(la identidad escalada por $p$). Para una normal unitaria cualquiera
$\vec n$:

$$\vec t = \boldsymbol\sigma\,\vec n = p\,\mathbf{I}\,\vec n = p\,\vec n.$$

El vector tracción es **paralelo a $\vec n$** y de magnitud $p$.

$$\sigma_n = \vec t \cdot \vec n = p\,(\vec n\cdot\vec n) = p$$

$$\vec\tau = \vec t - \sigma_n\,\vec n = p\,\vec n - p\,\vec n = \vec 0 \;\Rightarrow\; |\vec\tau| = 0$$

Por tanto, **sobre todo plano**: $\sigma_n = p$ y $|\vec\tau| = 0$. Todos
los planos son planos principales.

**2)** En la app:

- Los **tres círculos degeneran en un único punto** $(p, 0)$ porque
  $\sigma_1 = \sigma_2 = \sigma_3 = p$.
- Los tres puntos $x', y', z'$ caen en ese mismo punto, sin moverse
  cuando giras los ángulos.
- Cualquier plano oblicuo da la misma estrella magenta en $(p, 0)$.

Esta es **la única configuración tensional** en la que rotar el cubo no
afecta a las tensiones. En la práctica, un sólido sumergido en un fluido
en reposo está en estado hidrostático (con $p$ negativa, compresión).
"""
    )

