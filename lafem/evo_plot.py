
from itertools import product
from matplotlib import pyplot as plt

from sucrose.post import plot_evolution, LogDataFrame


def main(case: str):
    try:
        df = LogDataFrame(
            "./lafem/",
            runs=[case],
            tags=["loss(train)", "loss(validate)"]
        ).load()
    except ValueError:
        return

    fig = plt.figure()
    axes = fig.add_subplot()
    plot_evolution(
        df,
        axes,
        {(case, tag): val for tag, val in zip(
            ["loss(train)", "loss(validate)"],
            ["r-", "g-"]
        )},
        xlabel="step",
        ylabel="loss",
        log_scale="xy",
        legend_fmt="{tag}"
    )
    axes.set_title(case)
    st = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2]
    axes.set_yticks(st, [str(s) for s in st])
    axes.grid(True)
    axes.legend()
    fig.tight_layout()

    fig.savefig("./lafem/figures/{}.png".format(case.replace('/', '_')))
    plt.close(fig)


if __name__ == "__main__":
    for data_type, noise_type, model_type in product(
        ["equtri2", "gau3", "gau2", "tri1", "tri2"],
        ["ln01", "ln05", "nn"],
        ["multi", "single", "nograd"]
    ):
        case = "{}/unet100_{}_{}".format(data_type, noise_type, model_type)
        main(case)
