import pandas as pd
from matplotlib import pyplot as plt

from src.constants import RAIN_THRESH


def plot_rainfall(df: pd.DataFrame):
    fig, ax = plt.subplots(dpi=200, figsize=(8, 4))

    df["name_index"] = pd.factorize(df["name"])[0]

    imerg_pivot = df.pivot_table(
        index="name_index",
        columns="valid_date",
        values="mean",
        aggfunc="sum",
        fill_value=0,
    )

    imerg_pivot.plot(kind="bar", stacked=True, ax=ax, cmap="Paired")

    middle_date = imerg_pivot.columns[1]

    for i, row in imerg_pivot.iterrows():
        total = row.sum()
        ax.text(
            i,
            total + 2,
            f"{total:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    ax.axhline(RAIN_THRESH, color="crimson", linestyle="--")

    ax.annotate(
        f"Seuil : {RAIN_THRESH} mm",
        xy=(-0.2, RAIN_THRESH),
        fontsize=10,
        color="crimson",
        ha="left",
        va="bottom",
    )

    ax.set_xticks(range(len(imerg_pivot)))
    ax.set_xticklabels(df["name"].unique(), rotation=90)
    ax.legend(title="Date")

    ax.set_xlabel("Région")
    ax.set_ylabel(
        f"Précipitations totales sur trois jours,\n"
        f"moyenne sur région, centrées sur {middle_date} (mm)"
    )
    ax.set_title(f"Précipitations sur trois jours, centrées sur {middle_date}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax
