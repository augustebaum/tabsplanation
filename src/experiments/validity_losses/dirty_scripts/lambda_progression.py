r"""Plot the loss progression as \lambda changes."""

import matplotlib.pyplot as plt
import pandas as pd

from experiments.shared.utils import load_mpl_style

plt.style.use("tableau-colorblind10")
load_mpl_style()
df = pd.read_csv("results.csv", header=[0, 1], index_col=0)
fig = df.plot(y="mean", yerr="sem").get_figure()
ax = fig.axes[0]
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("Validity rate (\%)")
fig.savefig("logit_source_lambda_progression.svg")
