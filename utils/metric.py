import torchio as tio
from pathlib import Path
import torch
import numpy as np
import copy
from monai.metrics import compute_hausdorff_distance


def one_hot_encode(array):
    return np.eye(2)[array].transpose(0, 4, 1, 2, 3)


def metric(gt, pred, spacing=None):
    # * input shape: (batch, channel, height, width)
    preds = pred.detach().numpy()
    gts = gt.detach().numpy()
    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)  # float data does not support bit_and and bit_or

    # if spacing:
    #     # pred = pred[None, ::]
    #     # gdth = gdth[None, ::]
    #     pred_onehot = one_hot_encode(pred)
    #     gdth_onehot = one_hot_encode(gdth)
    #     hs95 = compute_hausdorff_distance(pred_onehot, gdth_onehot, percentile=95).numpy()
    #     hs95 = hs95[0][0]
    # else:
    hs95 = 0

    gdth = gdth.squeeze()
    pred = pred.squeeze()

    fp_array = copy.deepcopy(pred)  # keep pred unchanged
    fn_array = copy.deepcopy(gdth)
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = gdth & pred
    union = gdth | pred
    intersection_sum = np.count_nonzero(intersection)
    union_sum = np.count_nonzero(union)

    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp < 1] = 0

    tmp2 = gdth - pred
    fn_array[tmp2 < 1] = 0

    tn_array = np.ones(gdth.shape) - union

    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

    smooth = 0.001
    precision = tp / (pred_sum + smooth)
    recall = tp / (gdth_sum + smooth)
    # specificity = tn / (tn + fp + smooth)
    #
    # false_positive_rate = fp / (fp + tn + smooth)
    # false_negtive_rate = fn / (fn + tp + smooth)
    #
    # jaccard = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

    # return jaccard, dice
    return precision, recall, dice, hs95
