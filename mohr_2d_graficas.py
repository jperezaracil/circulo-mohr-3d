"""
Funciones de dibujo para la página 2D YZ.

Dos paneles:
  - dibujar_elemento_2d(ax, sy, sz, tyz, theta_deg, plano)
        Cuadrado en el plano YZ, girado, con flechas de tensión normal
        (rojo) y cortante (azul) sobre cada cara. Si se pasa `plano`
        (resultado de `tensiones_en_plano_2d`), también se dibuja la
        línea del plano de corte y el vector tracción.

  - dibujar_mohr_2d(ax, sy, sz, tyz, theta_deg, plano)
        Círculo de Mohr 2D para el estado tensional, con los puntos
        Y' y Z' del elemento girado y, opcionalmente, la estrella del
        plano de corte.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mohr_2d_calculos import transformar_2d, principales_2d


def _flecha(ax, x0, y0, dx, dy, color, lw=2.0, ls='-'):
    ax.annotate('',
                xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color,
                                lw=lw, mutation_scale=14, linestyle=ls))


# ---------------------------------------------------------------------------
# Panel izquierdo: elemento físico
# ---------------------------------------------------------------------------
def dibujar_elemento_2d(ax, sy, sz, tyz, theta_deg, plano=None):
    ax.clear()
    syp, szp, typ = transformar_2d(sy, sz, tyz, theta_deg)
    th = np.deg2rad(theta_deg)

    # Vectores de los ejes rotados
    ey = np.array([ np.cos(th), np.sin(th)])     # eje y'
    ez = np.array([-np.sin(th), np.cos(th)])     # eje z'

    L = 1.0  # semilado del cuadrado

    # --- Cuadrado de referencia (sin girar) ---
    v0 = np.array([[L, L], [-L, L], [-L, -L], [L, -L], [L, L]])
    ax.plot(v0[:, 0], v0[:, 1], '--', color='gray', lw=0.8, alpha=0.5)
    ax.text(L + 0.08, 0, 'cara y', color='gray', fontsize=8,
            ha='left', va='center', alpha=0.7)
    ax.text(0, L + 0.08, 'cara z', color='gray', fontsize=8,
            ha='center', va='bottom', alpha=0.7)

    # --- Cuadrado girado ---
    v = np.array([
         L * ey + L * ez,
        -L * ey + L * ez,
        -L * ey - L * ez,
         L * ey - L * ez,
         L * ey + L * ez,
    ])
    ax.plot(v[:, 0], v[:, 1], 'k-', lw=1.6)
    ax.fill(v[:-1, 0], v[:-1, 1], color='#cfe2ff', alpha=0.4)

    # Centros de cada cara del cuadrado girado
    c_yp, c_yn =  L * ey, -L * ey
    c_zp, c_zn =  L * ez, -L * ez

    # Escala visual de las flechas
    smax = max(abs(sy), abs(sz), abs(tyz), 1e-9)
    k = 0.6 / smax

    # Tensiones normales (rojo)
    _flecha(ax, *c_yp, *(syp * k * ey), 'red')
    _flecha(ax, *c_yn, *(-syp * k * ey), 'red')
    _flecha(ax, *c_zp, *(szp * k * ez), 'red')
    _flecha(ax, *c_zn, *(-szp * k * ez), 'red')

    # Tensiones cortantes (azul)
    _flecha(ax, *c_yp, *(typ * k * ez), 'blue')
    _flecha(ax, *c_yn, *(-typ * k * ez), 'blue')
    _flecha(ax, *c_zp, *(typ * k * ey), 'blue')
    _flecha(ax, *c_zn, *(-typ * k * ey), 'blue')

    # --- Plano de corte (si está activo) ---
    if plano is not None:
        n    = plano["n"]
        tang = plano["tang"]
        # Línea del plano (en 2D el "plano" es una recta)
        size = 1.8 * L
        p1 = -size * tang
        p2 =  size * tang
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='magenta', lw=2.2, alpha=0.6)

        # Normal del plano
        _flecha(ax, 0, 0, *(1.6 * n), color='darkgreen', lw=1.5)
        ax.text(*(1.75 * n), r'$\vec{n}$',
                color='darkgreen', fontsize=11, fontweight='bold')

        # Vector tracción
        t_vec   = plano["t"]
        sigma_n = plano["sigma_n"]
        tau_vec = plano["tau_vec"]
        _flecha(ax, 0, 0, *(t_vec * k), color='purple', lw=2.5)
        ax.text(*(t_vec * k * 1.1), r'$\vec{t}$',
                color='purple', fontsize=11, fontweight='bold')

        # Descomposición de t (líneas discontinuas)
        _flecha(ax, 0, 0, *(sigma_n * k * n),
                color='#c0392b', lw=1.2, ls='--')
        if np.linalg.norm(tau_vec) > 1e-6 * smax:
            _flecha(ax, 0, 0, *(tau_vec * k),
                    color='#1a5276', lw=1.2, ls='--')

    # --- Ejes globales y, z; eje x sale del plano ---
    ax.annotate('', xy=(2.6, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax.annotate('', xy=(0, 2.6), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax.text(2.7, 0, 'y', color='gray', va='center')
    ax.text(0, 2.7, 'z', color='gray', ha='center')
    # Indicador "x sale del plano" (círculo con punto, convención ⊙)
    circle = plt.Circle((-2.4, 2.4), 0.12, color='gray', fill=False, lw=1.2)
    ax.add_patch(circle)
    ax.plot(-2.4, 2.4, 'o', color='gray', markersize=3)
    ax.text(-2.2, 2.4, 'x (sale del plano)',
            color='gray', fontsize=8, va='center')

    # Indicador del giro θ
    if abs(theta_deg) > 0.5:
        arc = mpatches.Arc((0, 0), 0.9, 0.9, angle=0,
                           theta1=min(0, theta_deg),
                           theta2=max(0, theta_deg),
                           color='green', lw=1.5)
        ax.add_patch(arc)
        ax.text(0.6 * np.cos(th / 2), 0.6 * np.sin(th / 2),
                f'$\\theta={theta_deg:.0f}°$',
                color='green', fontsize=10)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_title('Elemento en plano YZ  (eje X sale del plano)')
    ax.axhline(0, color='lightgray', lw=0.5, zorder=0)
    ax.axvline(0, color='lightgray', lw=0.5, zorder=0)


# ---------------------------------------------------------------------------
# Panel derecho: círculo de Mohr 2D
# ---------------------------------------------------------------------------
def dibujar_mohr_2d(ax, sy, sz, tyz, theta_deg, plano=None, consulta=None):
    """Círculo de Mohr 2D para la sección YZ.

    Convención: punto Y plotea en (σy', −τy'z'), punto Z en (σz', +τy'z').
    Así una rotación física antihoraria θ del elemento se ve también como
    rotación antihoraria 2θ del punto sobre el círculo.
    """
    ax.clear()
    s1, s2, R, C, theta_p = principales_2d(sy, sz, tyz)
    syp, szp, typ = transformar_2d(sy, sz, tyz, theta_deg)

    # Círculo
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(C + R * np.cos(th), R * np.sin(th), 'k-', lw=1.5)
    ax.plot(C, 0, 'k+', markersize=10)

    # Tensiones principales
    ax.plot([s1, s2], [0, 0], 'go', markersize=8)
    ax.text(s1,  -(0.08 * R + 0.6), f'$\\sigma_1={s1:.1f}$',
            color='green', ha='center', va='top', fontsize=10)
    ax.text(s2,   (0.08 * R + 0.6), f'$\\sigma_2={s2:.1f}$',
            color='green', ha='center', va='bottom', fontsize=10)

    # Estado inicial (θ = 0): puntos Y y Z más tenues
    ax.plot(sy, -tyz, 'o', color='lightcoral', markersize=7, alpha=0.5)
    ax.plot(sz,  tyz, 'o', color='lightblue',  markersize=7, alpha=0.5)
    ax.plot([sy, sz], [-tyz, tyz], '--', color='gray', lw=0.7, alpha=0.5)

    # Estado actual: puntos Y' y Z' con su diámetro
    ax.plot(syp, -typ, 'o', color='red',  markersize=10, zorder=5)
    ax.plot(szp,  typ, 'o', color='blue', markersize=10, zorder=5)
    ax.plot([syp, szp], [-typ, typ], '-',
            color='purple', lw=1.5, alpha=0.8)
    ax.text(syp, -typ, f"  Y' ({syp:.1f}, {-typ:+.1f})",
            color='red', va='center', fontsize=9)
    ax.text(szp,  typ, f"  Z' ({szp:.1f}, {typ:+.1f})",
            color='blue', va='center', fontsize=9)

    # Punto del plano de corte
    if plano is not None:
        sn   = plano["sigma_n"]
        tau  = plano["tau_signed"]
        ax.plot(sn, -tau, marker='*', color='magenta', markersize=20,
                markeredgecolor='black', markeredgewidth=0.7, zorder=6,
                label='Plano de corte')
        ax.text(sn, -tau, f"   Plano ({sn:.1f}, {-tau:+.1f})",
                color='magenta', fontsize=9, fontweight='bold', va='center')

    # Punto de consulta (σ_q, τ_q)
    if consulta is not None:
        sq  = consulta["sigma_q"]
        tq  = consulta["tau_q"]
        col = '#e67e22' if consulta["on_circle"] else '#7f8c8d'
        ax.plot(sq, tq, marker='D', color=col, markersize=12,
                markeredgecolor='black', markeredgewidth=0.7, zorder=7,
                label='Punto consulta')
        ax.text(sq, tq, f"   Consulta ({sq:.1f}, {tq:+.1f})",
                color=col, fontsize=9, fontweight='bold', va='center')
        # Si no está sobre el círculo, dibuja una línea hasta el punto
        # más cercano del círculo para mostrar la distancia.
        if not consulta["on_circle"]:
            r_q = consulta["radio_q"]
            if r_q > 1e-9:
                # Proyección sobre el círculo desde el centro C
                sx = C + (sq - C) * R / r_q
                ty =       tq      * R / r_q
                ax.plot([sq, sx], [tq, ty], ':', color=col, lw=1.2)
                ax.plot(sx, ty, 'o', color=col, markersize=6,
                        markeredgecolor='black', markeredgewidth=0.5)

    # Encuadre con margen
    margen = 0.3 * R + 1.0
    xmin = min(s2, 0.0) - margen
    xmax = max(s1, 0.0) + margen
    ymin = -R - margen
    ymax =  R + margen
    ax.axhline(0, color='gray', lw=0.7)
    ax.axvline(0, color='gray', lw=0.7)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$\tau$')
    ax.grid(alpha=0.3)
    ax.set_title(rf'Círculo de Mohr 2D · C={C:.1f} · R={R:.1f} '
                 rf'· $\theta_p$={theta_p:.1f}°')
