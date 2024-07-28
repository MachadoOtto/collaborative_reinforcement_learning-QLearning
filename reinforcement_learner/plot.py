from argparse import ArgumentParser

import config
import pandas as pd
from utils import plot_learning
from utils_plot import plot_learning_curve


def main(csv_path: str, segment: int | None = None) -> None:
    model_name = csv_path.split("/")[-1].split(".")[0]

    data = pd.read_csv(csv_path)
    if segment is not None:
        data = data[:segment]
    else:
        segment = len(data)

    plot_learning(
        data.index.values,
        data["cumulative_reward"],
        data["epsilon"],
        f"{config.OUT_DIR}/plots/{model_name}-{segment}.png",
    )

    plot_learning_curve(data["cumulative_reward"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to the csv file")
    parser.add_argument(
        "--segment", type=int, default=None, help="Max number of episodes to plot"
    )

    args = parser.parse_args()
    main(csv_path=args.csv, segment=args.segment)
