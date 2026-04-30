"""
Catálogo de materiales para análisis de pandeo.

Para cada material se almacenan:
    E      módulo de Young (MPa)
    fy     límite elástico / tensión de rotura por compresión (MPa)
    tetmajer   tupla (a, b) con las constantes de Tetmajer en MPa, o None
               si no aplican constantes tabuladas habituales para ese
               material. La fórmula es  σ_cr = a − b·λ.

Las constantes de Tetmajer son empíricas y dependen del material y de la
fuente bibliográfica. Las que se usan aquí son las habituales en la
literatura europea de Resistencia de Materiales:

    Acero suave (S235/S275):       a = 310,  b = 1.14   (MPa)
    Acero alta resistencia (S355):  a = 335,  b = 0.62
    Madera (pino):                  a = 28.7, b = 0.19
    Aluminio:                       (no se incluye Tetmajer estándar)

Si el alumno escoge "Personalizado", puede ajustar E y fy por sliders;
en ese caso no se dibuja la curva de Tetmajer.
"""

MATERIALES = {
    "Acero S235": {
        "E":  210_000.0,
        "fy": 235.0,
        "tetmajer": (310.0, 1.14),
    },
    "Acero S275": {
        "E":  210_000.0,
        "fy": 275.0,
        "tetmajer": (310.0, 1.14),
    },
    "Acero S355": {
        "E":  210_000.0,
        "fy": 355.0,
        "tetmajer": (335.0, 0.62),
    },
    "Aluminio 6061-T6": {
        "E":   70_000.0,
        "fy": 240.0,
        "tetmajer": None,
    },
    "Madera C24 (pino)": {
        "E":   11_000.0,
        "fy":  21.0,
        "tetmajer": (28.7, 0.19),
    },
    "Personalizado": {
        "E":  210_000.0,
        "fy": 275.0,
        "tetmajer": None,
    },
}
