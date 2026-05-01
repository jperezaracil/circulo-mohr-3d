"""
Cálculos de tensiones planas en la sección YZ.

Convenciones:
    El eje longitudinal de la pieza es X.
    El plano analizado es YZ (la sección transversal del sólido).
    Las componentes activas son σy, σz, τyz.
    El ángulo θ es la rotación del elemento alrededor del eje X
    (rotación en el plano YZ, sentido antihorario mirando desde +X).

Convención del círculo de Mohr 2D
---------------------------------
Para que una rotación física antihoraria de θ del elemento se vea
también como rotación antihoraria de 2θ sobre el círculo (la
convención más pedagógica), el punto correspondiente a la cara y' se
plotea en (σy', -τy'z') y el de la cara z' en (σz', +τy'z'). Con esta
convención los puntos x' (en 3D, el equivalente a y') siempre quedan
en lados diametralmente opuestos del círculo.
"""

import numpy as np


def transformar_2d(sy, sz, tyz, theta_deg):
    """
    Tensiones en un elemento girado theta grados respecto al sistema YZ.
    Devuelve (σy', σz', τy'z').
    """
    th = np.deg2rad(theta_deg)
    c2, s2 = np.cos(2 * th), np.sin(2 * th)
    media   = (sy + sz) / 2.0
    semidif = (sy - sz) / 2.0
    syp =  media + semidif * c2 + tyz * s2
    szp =  media - semidif * c2 - tyz * s2
    typ = -semidif * s2 + tyz * c2
    return float(syp), float(szp), float(typ)


def principales_2d(sy, sz, tyz):
    """
    Tensiones principales en el plano YZ y ángulo principal.
    Devuelve (σ1, σ2, R, C, θp) donde C es la abscisa del centro
    del círculo, R su radio, y θp el ángulo (en grados) que orienta
    la dirección principal mayor σ1 medido respecto al eje y.
    """
    media = (sy + sz) / 2.0
    R = float(np.sqrt(((sy - sz) / 2.0) ** 2 + tyz ** 2))
    s1 = float(media + R)
    s2 = float(media - R)
    theta_p = 0.5 * float(np.degrees(np.arctan2(2.0 * tyz, sy - sz)))
    return s1, s2, R, float(media), theta_p


def tensiones_en_plano_2d(sy, sz, tyz, alpha_deg):
    """
    Tensiones sobre un plano de corte cuya normal forma un ángulo
    alpha con el eje y (en el plano YZ, sentido antihorario mirando
    desde +X).

    Devuelve un dict con:
        n           Vector unitario normal al plano (en YZ)
        tang        Vector unitario tangente al plano (perpendicular a n,
                    en sentido antihorario)
        t           Vector tracción t = σ·n  (en YZ)
        sigma_n     Tensión normal sobre el plano
        tau_signed  Tensión cortante con signo (positiva si τ apunta en
                    el sentido tangente antihorario respecto a n)
        tau_mag     |τ|, magnitud de la cortante
    """
    a = np.deg2rad(alpha_deg)
    n    = np.array([np.cos(a),  np.sin(a)])
    tang = np.array([-np.sin(a), np.cos(a)])

    # Tensor 2D (en el plano YZ)
    T = np.array([[sy,  tyz],
                  [tyz, sz ]], dtype=float)
    t = T @ n
    sigma_n   = float(np.dot(t, n))
    tau_vec   = t - sigma_n * n
    tau_signed = float(np.dot(tau_vec, tang))
    tau_mag   = float(np.linalg.norm(tau_vec))

    return {
        "n": n,
        "tang": tang,
        "t": t,
        "sigma_n": sigma_n,
        "tau_vec": tau_vec,
        "tau_signed": tau_signed,
        "tau_mag": tau_mag,
    }
