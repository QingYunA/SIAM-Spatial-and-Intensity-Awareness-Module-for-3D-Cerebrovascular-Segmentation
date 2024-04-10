import numpy as np

# import scipy.signal
from collections import Counter
from scipy.ndimage import maximum_filter
import SimpleITK as sitk
from pathlib import Path
import os
import copy
from tqdm import tqdm, trange
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
import rich


def create_progress(bar_width=40):
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )
    return progress


# * 设置属性


def set_properties(new_image, image):
    new_image.SetSpacing(image.GetSpacing())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetDirection(image.GetDirection())
    return new_image


def counter(label_array):
    count = Counter(label_array.flatten())
    zero_num = count[0]
    one_num = count[1]
    ratio = one_num / (zero_num + one_num)
    rich.print("血管占比:", ratio)
    return ratio


def rescale(data_path, save_path):
    data_list = sorted(Path(data_path).glob("*.mhd"))
    os.makedirs(save_path, exist_ok=True)
    progress = create_progress()
    # for (i, label) in progress.track(zip(data_list, label_list), total=len(data_list), description='Processing Data'):
    for i in progress.track(data_list, total=len(data_list), description="Processing Data"):
        progress.start()
        image = sitk.ReadImage(i)
        image = sitk.Cast(image, sitk.sitkUInt16)

        # rescale image
        rescale_filter = sitk.RescaleIntensityImageFilter()
        rescale_filter.SetOutputMaximum(255)
        rescale_filter.SetOutputMinimum(0)
        image = rescale_filter.Execute(image)
        print(save_path + i.name)

        sitk.WriteImage(image, save_path + i.name)


def create_iam(data_path, label_path, save_path):
    data_list = sorted(Path(data_path).glob("*.mhd"))
    label_list = sorted(Path(label_path).glob("*.mhd"))
    os.makedirs(save_path, exist_ok=True)
    progress = create_progress()
    for i, label in progress.track(zip(data_list, label_list), total=len(data_list), description="Processing Data"):
        progress.start()
        image = sitk.ReadImage(i)
        # image = sitk.Cast(image, sitk.sitkUInt16)
        array = sitk.GetArrayFromImage(image)
        #
        label_image = sitk.ReadImage(label)
        label_array = sitk.GetArrayFromImage(label_image)
        ratio = counter(label_array)
        #
        flatten_array = np.sort(array.flatten())
        array_len = len(flatten_array)
        threshold = int(flatten_array[int((1 - ratio) * array_len)])
        rich.print("this data label ratio :", threshold)
        image = sitk.Threshold(image, lower=0, upper=threshold, outsideValue=0)
        # rescale image
        rescale_filter = sitk.RescaleIntensityImageFilter()
        rescale_filter.SetOutputMaximum(255)
        rescale_filter.SetOutputMinimum(0)
        image = rescale_filter.Execute(image)

        print(save_path + i.name)

        sitk.WriteImage(image, save_path + i.name)


if __name__ == "__main__":
    #
    # train source rescale
    # data_path = "/nvme/PCA/train/source/"
    # save_path = "/nvme/PCA/train/rescale_source/"
    # rescale(data_path, save_path)
    #
    # test source rescale
    # data_path = "/nvme/PCA/test/source/"
    # save_path = "/nvme/PCA/test/rescale_source/"
    # rescale(data_path, save_path)
    #
    # iam data produce
    data_path = "/nvme/PCA/train/rescale_source/"
    label_path = "/nvme/PCA/train/label/"
    save_path = "/nvme/PCA/IAM_data/"
    create_iam(data_path, label_path, save_path)
    #
