import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd


def plot_results(file="path/to/results.csv", dir="", best=True):
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    files = list(save_dir.glob("results*.csv"))
    assert len(
        files
    ), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."

    nrs = []
    for f in files:
        nrs.append(len(pd.read_csv(f).values[0]) - 2)
    fig, ax = plt.subplots(1, max(nrs), figsize=(3 * max(nrs), 4), tight_layout=True)
    ax = ax.ravel()

    for _, f in enumerate(files):
        # try:
            data = pd.read_csv(f)
            index = np.argmax(data.values[:, 4:-1].mean(-1))
            s = [x.strip() for x in data.columns]
            nr = len(data.values[0])
            x = data.values[:, 0]
            for i, j in enumerate(list(range(2, nr))):
                y = data.values[:, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=2)
                if best:
                    # best
                    ax[i].scatter(
                        index,
                        y[index],
                        color="r",
                        label=f"best:{index}",
                        marker="*",
                        linewidth=3,
                    )
                    ax[i].set_title(s[j] + f"\n{round(y[index], 5)}")
                else:
                    # last
                    ax[i].scatter(
                        x[-1], y[-1], color="r", label="last", marker="*", linewidth=3
                    )
                    ax[i].set_title(s[j] + f"\n{round(y[-1], 5)}")
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        # except Exception as e:
            # print(f"Warning: Plotting error for {f}: {e}")
    ax[1].legend()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    plot_results(file='/home/laughing/code/MetricTrainer/runs/AdaFace/results.csv')
