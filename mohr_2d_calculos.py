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


def localizar_punto_2d(sy, sz, tyz, sigma_q, tau_q):
    """
    Dado un estado tensional 2D y un punto de consulta (σ_q, τ_q) en el
    plano de Mohr, determina si ese punto cae sobre el círculo y, si es
    así, la orientación α del plano cuya cara tiene esa tensión.

    El punto (σ_q, τ_q) se interpreta tal cual se ve en el diagrama de
    Mohr (con la convención de la app: τ positivo hacia arriba).

    Devuelve dict con:
        on_circle    True si (σ_q − C)² + τ_q² ≈ R² (tolerancia relativa)
        dist         Distancia del punto al círculo (|radio − r|)
        radio_q      sqrt((σ_q − C)² + τ_q²)
        C, R         centro y radio del círculo
        alpha_y      α (grados) de la **normal** del plano cuya cara y'
                     coincide con (σ_q, τ_q). Equivale al ángulo θ que
                     hay que girar el elemento para que la cara y'
                     marque ese punto.
        alpha_z      α + 90° (la cara opuesta z' del mismo elemento
                     marca el punto diametralmente opuesto).
        sigma_perp   Tensión normal sobre el plano perpendicular
                     (cara z' a la vez que la y' marca (σ_q, τ_q)).
        tau_perp     Tensión cortante (con signo) sobre ese plano
                     perpendicular.
    """
    media = (sy + sz) / 2.0
    R = float(np.sqrt(((sy - sz) / 2.0) ** 2 + tyz ** 2))
    C = float(media)
    r_q = float(np.sqrt((sigma_q - C) ** 2 + tau_q ** 2))
    dist = abs(r_q - R)
    tol = 1e-3 * R + 1e-6
    on_circle = bool(dist <= tol)

    # Convención del diagrama: la cara y' del elemento girado un ángulo
    # α se plotea en (σ_y'(α), −τ_y'z'(α)).  Desarrollando:
    #     σ_q − C =  p cos(2α) + q sin(2α)
    #     τ_q     =  p sin(2α) − q cos(2α)
    # con p=(σy−σz)/2, q=τyz.  Escribiendo (p, q) = R(cosβ, sinβ) y
    # (σ_q−C, τ_q) = R(cos ψ, sin ψ), resulta ψ = 2α − β, así que:
    p, q = (sy - sz) / 2.0, tyz
    psi  = np.arctan2(tau_q,     sigma_q - C)
    beta = np.arctan2(q,         p)
    alpha_y_rad = (psi + beta) / 2.0
    alpha_y = float(np.degrees(alpha_y_rad))
    # Normaliza a (−90°, 90°]
    while alpha_y >  90.0: alpha_y -= 180.0
    while alpha_y <= -90.0: alpha_y += 180.0
    alpha_z = alpha_y + 90.0
    if alpha_z > 90.0:
        alpha_z -= 180.0

    # Tensiones sobre la cara perpendicular (z' del mismo elemento)
    syp, szp, typ = transformar_2d(sy, sz, tyz, alpha_y)
    # Cuando la cara y' marca (σ_q, τ_q) en el diagrama (con τ_q en eje
    # vertical, convención app), la cara z' marca (σ_z', +τ_y'z') que
    # equivale a (σ_z', +τ).  La σ del plano perpendicular es szp y la
    # τ "tal y como se ve en el diagrama" en la cara z' es +typ.
    sigma_perp = float(szp)
    tau_perp   = float(typ)

    return {
        "on_circle": on_circle,
        "dist":      dist,
        "radio_q":   r_q,
        "C":         C,
        "R":         R,
        "alpha_y":   alpha_y,
        "alpha_z":   float(alpha_z),
        "sigma_perp": sigma_perp,
        "tau_perp":   tau_perp,
    }
