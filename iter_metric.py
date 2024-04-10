import os
from pathlib import Path
import pandas as pd


def generate_cvs(path, path_results):
    csv_list = Path(path).glob("*/*/*/*.csv")
    csv_list2 = Path(path_results).glob("*/*/*/*.csv")
    data_list = {}
    name, path_ls, precision, recall, dice = [], [], [], [], []
    for i in csv_list:
        name.append(i.parts[1])
        path_ls.append(i)
        df = pd.read_csv(i)

        for j in df.columns:
            if j == "dice_ls":
                dice.append(df.iloc[-1][j])
            elif j == "precision_ls":
                precision.append(df.iloc[-1][j])
            elif j == "recall_ls":
                recall.append(df.iloc[-1][j])
    if path_results:
        for i in csv_list2:
            df = pd.read_csv(i)

            for j in df.columns:
                if j == "jaccard":
                    break
                if j == "dice_ls":
                    dice.append(df.iloc[-1][j])
                    name.append(i.parts[1])
                    path_ls.append(i)
                elif j == "precision_ls":
                    precision.append(df.iloc[-1][j])
                elif j == "recall_ls":
                    recall.append(df.iloc[-1][j])
    data_list = {"name": name, "path": path_ls, "precision": precision, "recall": recall, "dice": dice}

    df = pd.DataFrame(data_list)
    df.to_csv("./all_metrics.csv")


if __name__ == "__main__":
    # network = "siamBETA"
    # path = f"./logs/{network}"
    # csv_list = Path(path).glob("*/*/*.csv")
    #
    # for i in csv_list:
    #     df = pd.read_csv(i)
    #     if "dice_ls" in df.columns:
    #         dice_mean = df.iloc[-1]["dice_ls"]
    #     else:
    #         dice_mean = df.iloc[-1]["dice"]
    #     print(i.parent, dice_mean)
    #
    # print("--------- results:")
    #
    # path = f"./results/{network}"
    # csv_list = Path(path).glob("*/*/*.csv")
    # csv_list = Path(path).glob("*/*/*.csv")
    #
    # for i in csv_list:
    #     df = pd.read_csv(i)
    #     if "dice_ls" in df.columns:
    #         dice_mean = df.iloc[-1]["dice_ls"]
    #     else:
    #         dice_mean = df.iloc[-1]["dice"]
    #     print(i.parent, dice_mean)
    path = "./logs"
    path_result = "./results"
    generate_cvs(path=path, path_results=path_result)
