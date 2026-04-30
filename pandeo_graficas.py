"""
Funciones de dibujo para la aplicación de pandeo.

Hay dos paneles:
  - dibujar_columna(ax, K, L, n_seguridad)
        Columna con sus apoyos y la deformada del primer modo.
  - dibujar_curvas(ax, lam_actual, sigma_actual, E, fy, tetmajer, lam_lim)
        Diagrama σ_cr vs λ con las cuatro fórmulas y el punto actual.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pandeo_calculos import (
    deformada_modo1,
    esbeltez_limite,
    sigma_cr_euler,
    sigma_cr_johnson,
    sigma_cr_tetmajer,
    sigma_cr_rankine,
)


# ---------------------------------------------------------------------------
# Apoyos
# ---------------------------------------------------------------------------
def _apoyo_articulado(ax, x, y, w, hacia_abajo=True):
    """Triángulo + suelo rayado (articulación)."""
    s = -1 if hacia_abajo else 1
    ax.fill([x - w/2, x + w/2, x],
            [y + s*w, y + s*w, y],
            color='lightgray', edgecolor='black', lw=1.2, zorder=3)
    ax.plot([x - w*1.4, x + w*1.4], [y + s*w, y + s*w],
            'k-', lw=1.2, zorder=3)
    for xx in np.linspace(x - w*1.3, x + w*1.3, 8):
        ax.plot([xx, xx + s*w*0.18],
                [y + s*w, y + s*w*1.45],
                'k-', lw=0.7, zorder=3)


def _apoyo_empotrado(ax, x, y, w, hacia_abajo=True):
    """Bloque rayado (empotramiento)."""
    s = -1 if hacia_abajo else 1
    rect = mpatches.Rectangle((x - w*1.4, y if not hacia_abajo else y - w*0.35),
                              w*2.8, w*0.35,
                              facecolor='lightgray', edgecolor='black',
                              lw=1.2, zorder=3)
    ax.add_patch(rect)
    for xx in np.linspace(x - w*1.3, x + w*1.3, 10):
        ax.plot([xx, xx + s*w*0.22],
                [y + s*w*0.35, y + s*w*0.65],
                'k-', lw=0.7, zorder=3)


def _carga_aplicada(ax, x, y, w, hacia_abajo=True):
    """Flecha de carga apuntando al extremo libre."""
    s = -1 if hacia_abajo else 1
    ax.annotate('', xy=(x, y), xytext=(x, y + s * w * 1.8),
                arrowprops=dict(arrowstyle='-|>', color='black',
                                lw=2, mutation_scale=18), zorder=4)
    ax.text(x + w*0.4, y + s*w*1.3, 'P',
            fontsize=12, fontweight='bold', va='center')


# ---------------------------------------------------------------------------
# Panel izquierdo: columna y deformada
# ---------------------------------------------------------------------------
def dibujar_columna(ax, K, L, n_seguridad=None):
    """
    Dibuja la columna con la deformada del primer modo. Si se pasa
    n_seguridad (= P_cr / P_aplicada), la amplitud y el color de la
    deformada cambian para reforzar visualmente el grado de seguridad.
    """
    ax.clear()

    s, x_def = deformada_modo1(K)
    y = s * L

    # Color y amplitud según margen de seguridad
    if n_seguridad is None or n_seguridad >= 2.0:
        amp = L * 0.04
        color = '#2ca02c'      # verde — seguro
    elif n_seguridad >= 1.0:
        amp = L * 0.12
        color = '#f1c40f'      # ámbar — al límite
    else:
        amp = L * 0.22
        color = '#e74c3c'      # rojo — pandea

    x = x_def * amp

    # Columna sin deformar (referencia)
    ax.plot([0, 0], [0, L], color='gray', lw=0.8, ls=':', alpha=0.6, zorder=1)
    # Columna deformada
    ax.plot(x, y, color=color, lw=3, zorder=2)

    # Apoyos según condiciones de contorno
    w = max(L * 0.045, 35.0)
    if abs(K - 1.0) < 1e-3:                  # biarticulada
        _apoyo_articulado(ax, 0, 0, w, hacia_abajo=True)
        _apoyo_articulado(ax, x[-1], L, w, hacia_abajo=False)
        _carga_aplicada(ax, x[-1], L + w*0.25, w, hacia_abajo=False)
    elif abs(K - 0.5) < 1e-3:                # biempotrada
        _apoyo_empotrado(ax, 0, 0, w, hacia_abajo=True)
        _apoyo_empotrado(ax, x[-1], L, w, hacia_abajo=False)
        _carga_aplicada(ax, x[-1], L + w*0.6, w, hacia_abajo=False)
    elif abs(K - 2.0) < 1e-3:                # empotrada-libre (voladizo)
        _apoyo_empotrado(ax, 0, 0, w, hacia_abajo=True)
        _carga_aplicada(ax, x[-1], L + w*0.25, w, hacia_abajo=False)
    elif abs(K - 0.7) < 1e-3:                # empotrada-articulada
        _apoyo_empotrado(ax, 0, 0, w, hacia_abajo=True)
        _apoyo_articulado(ax, x[-1], L, w, hacia_abajo=False)
        _carga_aplicada(ax, x[-1], L + w*0.25, w, hacia_abajo=False)

    # Texto explicativo
    ax.text(w*1.6, L*0.5,
            f"L = {L:.0f} mm\nK = {K}\nLₑ = K·L = {K*L:.0f} mm",
            fontsize=10, va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor='gray', alpha=0.9))

    # Encuadre
    margen_x = max(w * 1.8, abs(amp) * 1.6)
    ax.set_xlim(-margen_x*1.2, margen_x*2.5)
    ax.set_ylim(-w*1.6, L * 1.15)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_title('Columna y modo de pandeo')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Panel derecho: σ_cr vs λ con las cuatro curvas
# ---------------------------------------------------------------------------
def dibujar_curvas(ax, lam_actual, sigma_actual, E, fy, tetmajer):
    """Dibuja las cuatro curvas σ_cr-λ y marca el punto actual."""
    ax.clear()
    lam_lim = esbeltez_limite(E, fy)
    lam_max = max(2.5 * lam_lim, 1.5 * lam_actual, 200.0)
    lam = np.linspace(1.0, lam_max, 600)

    # Euler — toda la curva (gris fina cuando no es válida)
    se_full = sigma_cr_euler(lam, E)
    # Parte válida (λ ≥ λ_lim)
    mask_val = lam >= lam_lim
    ax.plot(lam[mask_val], se_full[mask_val],
            color='#1f77b4', lw=2.2, label='Euler', zorder=4)
    ax.plot(lam[~mask_val], se_full[~mask_val],
            color='#1f77b4', lw=1.0, ls=':', alpha=0.5, zorder=2)

    # Johnson — válida 0 ≤ λ ≤ λ_lim
    lam_j = np.linspace(0.0, lam_lim, 200)
    sj = sigma_cr_johnson(lam_j, E, fy)
    ax.plot(lam_j, sj, color='#ff7f0e', lw=2.2, label='Johnson', zorder=4)

    # Tetmajer — solo si el material lo soporta
    if tetmajer is not None:
        a, b = tetmajer
        lam_t = np.linspace(0.0, lam_lim, 200)
        st = sigma_cr_tetmajer(lam_t, a, b)
        st = np.clip(st, 0.0, fy)
        ax.plot(lam_t, st, color='#2ca02c', lw=2.2, label='Tetmajer', zorder=4)

    # Rankine-Gordon — toda la curva
    sr = sigma_cr_rankine(lam, E, fy)
    ax.plot(lam, sr, color='#d62728', lw=2.0, ls='--',
            label='Rankine-Gordon', zorder=4)

    # Línea de fluencia y esbeltez límite
    ax.axhline(fy, color='gray', lw=1.0, ls=':', alpha=0.7)
    ax.text(lam_max*0.99, fy*1.02, f'σy = {fy:.0f} MPa',
            ha='right', va='bottom', fontsize=9, color='gray')
    ax.axvline(lam_lim, color='gray', lw=1.0, ls=':', alpha=0.7)
    ax.text(lam_lim, fy*1.35, f'λlim = {lam_lim:.1f}',
            ha='center', va='top', fontsize=9, color='gray',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.85))

    # Punto del estado actual
    ax.plot(lam_actual, sigma_actual, 'o',
            color='black', markersize=12,
            markeredgecolor='yellow', markeredgewidth=2.2,
            zorder=6, label=f'Estado actual')
    ax.annotate(f'  ({lam_actual:.0f}, {sigma_actual:.0f} MPa)',
                xy=(lam_actual, sigma_actual),
                xytext=(8, 8), textcoords='offset points',
                fontsize=10, fontweight='bold')

    ax.set_xlabel(r'Esbeltez  $\lambda = L_e / r_{\min}$')
    ax.set_ylabel(r'Tensión crítica  $\sigma_{cr}$  (MPa)')
    ax.set_xlim(0.0, lam_max)
    ax.set_ylim(0.0, fy * 1.4)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(r'Curvas de pandeo  $\sigma_{cr}$ vs $\lambda$')
