import torch
import numpy as np
from monai.metrics import compute_hausdorff_distance


def save_csv(save_path, **kwargs):
    data = {}
    for k, v in kwargs.items():
        data[k] = v

    print(data)


if __name__ == "__main__":
    tmp_ls, tmp_ls2 = [1], [2]
    save_csv("test", tmp_ls=tmp_ls, tmp_ls2=tmp_ls2)
