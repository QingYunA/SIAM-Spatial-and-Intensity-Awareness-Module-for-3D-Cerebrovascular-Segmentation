import os
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    path = "./results/siam/"
    csv_list = Path(path).glob("*/*/*.csv")

    # *  find top 5 dice
    max_dice = 0
    max_csv = None
    for i in csv_list:
        df = pd.read_csv(i)
        dice = max(df["dice"])
        if dice > max_dice:
            max_dice = dice
            max_csv = i

    print(f"max_dice: {max_dice}, max_csv: {max_csv.parent}")
