import matplotlib.pyplot as plt

latex_preamble = r"""
\usepackage{libertine}
"""


def set_matplotlib_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": latex_preamble,
            "font.size": 12,
            "mathtext.fontset": "stix",
        }
    )
