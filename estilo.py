"""
Estilo visual común para todas las gráficas matplotlib de la app.

Importar este módulo aplica una paleta clara y discreta a las figuras
para que encajen con el tema claro de Streamlit. Usar:

    import estilo  # noqa: F401   (aplica los rcParams al importarse)

o bien, dentro de una función de dibujo:

    from estilo import aplicar_estilo
    aplicar_estilo()
"""

import matplotlib.pyplot as plt


PALETA = {
    "fondo":        "#ffffff",
    "fondo_axes":   "#fbfcfd",
    "borde":        "#cdd5df",
    "rejilla":      "#e3e8ee",
    "texto":        "#1f2530",
    "texto_suave":  "#6b7280",
    "azul":         "#0a5fa8",
    "rojo":         "#c0392b",
    "verde":        "#1f8a3d",
    "morado":       "#7a3eaa",
    "naranja":      "#e67e22",
    "amarillo":     "#fff3b0",
    "magenta":      "#c2185b",
    "celeste":      "#cfe2ff",
}


def aplicar_estilo():
    """Aplica una paleta clara a matplotlib (rcParams globales)."""
    plt.rcParams.update({
        "figure.facecolor": PALETA["fondo"],
        "savefig.facecolor": PALETA["fondo"],
        "axes.facecolor":   PALETA["fondo_axes"],
        "axes.edgecolor":   PALETA["borde"],
        "axes.labelcolor":  PALETA["texto"],
        "axes.titlesize":   11,
        "axes.titleweight": "regular",
        "axes.titlepad":    8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "grid.color":       PALETA["rejilla"],
        "grid.alpha":       0.7,
        "grid.linewidth":   0.7,
        "xtick.color":      PALETA["texto_suave"],
        "ytick.color":      PALETA["texto_suave"],
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "axes.labelsize":   10,
        "legend.frameon":   False,
        "legend.fontsize":  9,
        "font.family":      "sans-serif",
        "font.size":        10,
    })


# Aplica el estilo al importar el módulo.
aplicar_estilo()
