import os
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    network = "csrnet0.5"
    path = f"./logs/{network}"
    csv_list = Path(path).glob("*/*/*.csv")

    for i in csv_list:
        df = pd.read_csv(i)
        if "dice_ls" in df.columns:
            dice_mean = df.iloc[-1]["dice_ls"]
        else:
            dice_mean = df.iloc[-1]["dice"]
        print(i.parent, dice_mean)

    print("--------- results:")

    path = f"./results/{network}"
    csv_list = Path(path).glob("*/*/*.csv")
    csv_list = Path(path).glob("*/*/*.csv")

    for i in csv_list:
        df = pd.read_csv(i)
        if "dice_ls" in df.columns:
            dice_mean = df.iloc[-1]["dice_ls"]
        else:
            dice_mean = df.iloc[-1]["dice"]
        print(i.parent, dice_mean)
