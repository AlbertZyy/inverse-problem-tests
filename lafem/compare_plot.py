
from itertools import product
from matplotlib import pyplot as plt

from sucrose.post import plot_evolution, LogDataFrame


def main(cases: list[str], data_type: str):
    try:
        dfs = LogDataFrame(
            "./lafem/",
            runs=cases,
            tags=["loss(validate)"]
        ).load()
    except ValueError:
        return

    dfs = {
        key: df.rolling(10, min_periods=1).mean()
        for key, df in dfs.items()
    }

    fig = plt.figure()
    axes = fig.add_subplot()
    plot_evolution(
        dfs,
        axes=axes,
        fmts={(run, "loss(validate)"): val for run, val in zip(
            cases,
            ["r-", "g-", "b-", "r--", "g--", "b--", "r:", "g:", "b:"]
        )},
        xlabel="step",
        ylabel="validation loss",
        log_scale="y",
        legend_fmt="{run}"
    )
    axes.set_title(data_type)
    st = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2]
    axes.set_yticks(st, [str(s) for s in st])
    axes.grid(True)
    axes.legend()
    axes.set_xlim(5000, 30000)
    fig.tight_layout()

    fig.savefig("./lafem/figures/comp_{}.png".format(data_type))
    plt.close(fig)


if __name__ == "__main__":
    for data_type in ["equtri2", "gau2", "gau3", "tri1", "tri2"]:
        cases = []
        for noise_type, model_type in product(
            ["nn", "ln01", "ln05"],
            ["multi", "single", "nograd"]
        ):
            case = "{}/unet100_{}_{}".format(data_type, noise_type, model_type)
            cases.append(case)

        main(cases, data_type)
