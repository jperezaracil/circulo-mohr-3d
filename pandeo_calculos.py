"""
Cálculos de pandeo de columnas.

Reúne:
  - Propiedades de la sección (área, inercias, radios de giro)
    para perfiles de catálogo y para secciones paramétricas.
  - Las cuatro fórmulas clásicas de tensión crítica:
      Euler          σ = π² E / λ²
      Johnson        σ = σ_y − (σ_y² / 4π² E) λ²
      Tetmajer       σ = a − b λ              (constantes empíricas)
      Rankine-Gordon σ = σ_y / (1 + (σ_y / π² E) λ²)
  - Cálculo de longitud de pandeo Lₑ = K·L según condición de contorno.
  - Forma del primer modo de pandeo (deformada) para cada caso de apoyos.

Unidades: tensiones en MPa, longitudes en mm, áreas en mm², inercias en mm⁴.
"""

import numpy as np

from pandeo_perfiles import PERFILES


# ---------------------------------------------------------------------------
# Condiciones de contorno (factor K = Lₑ / L)
# ---------------------------------------------------------------------------
CONDICIONES = {
    "Biarticulada":               {"K": 1.0,  "modo": "pinned-pinned"},
    "Biempotrada":                {"K": 0.5,  "modo": "fixed-fixed"},
    "Empotrada-libre (voladizo)": {"K": 2.0,  "modo": "fixed-free"},
    "Empotrada-articulada":       {"K": 0.7,  "modo": "fixed-pinned"},
}


# ---------------------------------------------------------------------------
# Propiedades de la sección
# ---------------------------------------------------------------------------
def propiedades_seccion(tipo, params):
    """
    Devuelve (A, Iy, Iz, h, b) según el tipo de sección.

    Iy es la inercia mayor (eje fuerte) e Iz la menor (eje débil).
    Para cálculo de pandeo se usará I_min = Iz (la sección pandea
    alrededor del eje de menor inercia).
    """
    if tipo == "Catálogo":
        d = PERFILES[params["nombre"]]
        Iy, Iz = d["Iy"], d["Iz"]
        return d["A"], max(Iy, Iz), min(Iy, Iz), d["h"], d["b"]

    if tipo == "Rectangular maciza":
        b, h = float(params["b"]), float(params["h"])
        A = b * h
        I_b = b * h**3 / 12.0    # alrededor del eje paralelo a b
        I_h = h * b**3 / 12.0    # alrededor del eje paralelo a h
        return A, max(I_b, I_h), min(I_b, I_h), h, b

    if tipo == "Circular maciza":
        D = float(params["D"])
        A = np.pi * D**2 / 4.0
        I = np.pi * D**4 / 64.0
        return A, I, I, D, D

    if tipo == "Tubo circular":
        De, t = float(params["De"]), float(params["t"])
        Di = max(De - 2.0 * t, 0.0)
        A = np.pi * (De**2 - Di**2) / 4.0
        I = np.pi * (De**4 - Di**4) / 64.0
        return A, I, I, De, De

    if tipo == "Tubo rectangular":
        b, h, t = float(params["b"]), float(params["h"]), float(params["t"])
        bi = max(b - 2.0 * t, 0.0)
        hi = max(h - 2.0 * t, 0.0)
        A   = b * h - bi * hi
        I_b = (b * h**3 - bi * hi**3) / 12.0
        I_h = (h * b**3 - hi * bi**3) / 12.0
        return A, max(I_b, I_h), min(I_b, I_h), h, b

    raise ValueError(f"Tipo de sección desconocido: {tipo}")


def radios_de_giro(A, Iy, Iz):
    """Devuelve (r_y, r_z, r_min) en mm."""
    r_y = float(np.sqrt(Iy / A))
    r_z = float(np.sqrt(Iz / A))
    return r_y, r_z, min(r_y, r_z)


# ---------------------------------------------------------------------------
# Esbeltez y esbeltez límite
# ---------------------------------------------------------------------------
def esbeltez(L, K, r_min):
    """Esbeltez λ = Lₑ / r_min, donde Lₑ = K·L."""
    Le = K * L
    return Le, Le / r_min


def esbeltez_limite(E, fy):
    """λ_lim = π·√(E/fy) — frontera entre régimen elástico y plástico."""
    return np.pi * np.sqrt(E / fy)


# ---------------------------------------------------------------------------
# Las cuatro fórmulas de tensión crítica
# ---------------------------------------------------------------------------
def sigma_cr_euler(lam, E):
    """Hipérbola de Euler.  Devuelve nan en λ=0 (asíntota)."""
    lam = np.asarray(lam, dtype=float)
    out = np.full_like(lam, np.nan)
    mask = lam > 1.0e-9
    out[mask] = np.pi**2 * E / lam[mask]**2
    return out


def sigma_cr_johnson(lam, E, fy):
    """Parábola de Johnson, tangente a Euler en λ_lim, vale fy en λ=0."""
    lam = np.asarray(lam, dtype=float)
    return fy - (fy**2 / (4.0 * np.pi**2 * E)) * lam**2


def sigma_cr_tetmajer(lam, a, b):
    """Recta empírica de Tetmajer: σ_cr = a − b λ."""
    lam = np.asarray(lam, dtype=float)
    return a - b * lam


def sigma_cr_rankine(lam, E, fy):
    """Rankine-Gordon: σ_cr = fy / (1 + (fy/π² E) λ²). Válida para todo λ."""
    lam = np.asarray(lam, dtype=float)
    return fy / (1.0 + (fy / (np.pi**2 * E)) * lam**2)


def regimen_y_sigma(lam, E, fy):
    """
    Devuelve (σ_cr, nombre_regimen) usando el criterio Euler/Johnson:
    Euler en régimen elástico (λ ≥ λ_lim), Johnson en plástico.
    """
    lam_lim = esbeltez_limite(E, fy)
    if lam >= lam_lim:
        sigma = float(sigma_cr_euler(np.array([lam]), E)[0])
        return sigma, "Elástico (Euler)"
    sigma = float(sigma_cr_johnson(np.array([lam]), E, fy)[0])
    return sigma, "Plástico (Johnson)"


# ---------------------------------------------------------------------------
# Forma del primer modo de pandeo
# ---------------------------------------------------------------------------
def deformada_modo1(K, n_points=200):
    """
    Devuelve (s, x_norm) con s = y/L ∈ [0, 1] y x_norm la deformada
    normalizada a máximo 1, según las condiciones de contorno.

    K=1.0  biarticulada           media onda de seno
    K=0.5  biempotrada            onda cosenoidal completa
    K=2.0  empotrada-libre        cuarto de coseno
    K=0.7  empotrada-articulada   eigenfunción tan(βL)=βL, con βL≈4.4934
    """
    s = np.linspace(0.0, 1.0, n_points)

    if abs(K - 1.0) < 1.0e-3:
        x = np.sin(np.pi * s)
    elif abs(K - 0.5) < 1.0e-3:
        x = 0.5 * (1.0 - np.cos(2.0 * np.pi * s))
    elif abs(K - 2.0) < 1.0e-3:
        x = 1.0 - np.cos(np.pi * s / 2.0)
    elif abs(K - 0.7) < 1.0e-3:
        beta = 4.4934
        x = np.sin(beta * s) - beta * np.cos(beta * s) - beta * s + beta
        x = x / np.max(np.abs(x))
    else:
        x = np.sin(np.pi * s)
    return s, x
