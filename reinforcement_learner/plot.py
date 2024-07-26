import os

import pandas as pd
from utils import plot_learning

OUT_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/outputs"


def main(csv_path: str, segment: int | None = None) -> None:
    model_name = csv_path.split("/")[-1].split(".")[0]

    data = pd.read_csv(csv_path)
    if segment is not None:
        data = data[:segment]
    else:
        segment = len(data)

    plot_learning(
        data.index.values,
        data["score"],
        data["epsilon"],
        f"{OUT_DIR}/plots/{model_name}-{segment}.png",
    )


if __name__ == "__main__":
    file_path = f"{OUT_DIR}/stats/LunarLander-v2-version4.csv"
    main(file_path)
