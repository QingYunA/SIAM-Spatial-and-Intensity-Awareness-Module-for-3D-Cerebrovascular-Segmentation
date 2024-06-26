from pathlib import Path
from torchio.transforms import (
    RandomFlip,
    RandomSwap,
    RescaleIntensity,
    RandomNoise,
    ZNormalization,
    OneOf,
    Compose,
)
import os
import numpy as np
from torchio.data import UniformSampler, LabelSampler
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
import torchio
from torchio import AFFINE, DATA
import torchio as tio
import torch
import sys
from collections import Counter
import hydra
import tqdm
from rich.progress import track


sys.path.append("./")


def get_subjects(config):
    """
    @description: get the subjects for normal training
    """
    subjects = []
    if "predict" in config.job_name:
        img_path = Path(config.pred_data_path)
        gt_path = Path(config.pred_gt_path)
    else:
        img_path = Path(config.data_path)
        gt_path = Path(config.gt_path)
    dark_path = Path(config.dark_path)
    x_generator = sorted(img_path.glob(f"*{config.extension}"))
    gt_generator = sorted(gt_path.glob(f"*{config.extension}"))
    dark_generator = sorted(dark_path.glob(f"*{config.extension}"))

    ratio = config.dataset_ratio

    x_list = list(x_generator)
    gt_list = list(gt_generator)
    dark_list = list(dark_generator)

    if "train" in config.job_name:
        x_list = x_list[: int(len(x_list) * ratio)]
        gt_list = gt_list[: int(len(gt_list) * ratio)]
        dark_list = dark_list[: int(len(dark_list) * ratio)]

    for i, (source, gt, dark) in track(
        enumerate(zip(x_list, gt_list, dark_list)),
        description="[green]Preparing Dataset",
        total=len(x_generator),
    ):
        if "predict" in config.job_name:
            subject = tio.Subject(source=tio.ScalarImage(source), gt=tio.LabelMap(gt))
        else:
            # source_img = tio.ScalarImage(source)
            # gt_img = tio.LabelMap(gt)
            # source_ary = source_img.data.numpy()
            # gt_ary = gt_img.data.numpy()
            #
            # pos = np.sum(gt_ary == 1)
            #
            # threshold = np.sort(source_ary.flatten())[::-1][pos - 1]
            # # source_ary[source_ary > threshold] = threshold
            # source_ary[source_ary > threshold] = 0
            # source_ary = torch.tensor(source_ary)

            subject = tio.Subject(
                source=tio.ScalarImage(source),
                spatial_source=tio.ScalarImage(source),
                # intensity_source=ScalarImage(tensor=source_ary, affine=source_img.affine),
                intensity_source=ScalarImage(dark),
                gt=tio.LabelMap(gt),
                siam_gt=tio.ScalarImage(source),
            )
            # save test

            # break
        subjects.append(subject)
    return subjects


class SIAMDataLoader(torch.utils.data.Dataset):
    def __init__(self, config):
        self.subjects = []

        queue_length = 10
        samples_per_volume = 10

        self.swap_size = self.to_list(config.swap_size)
        self.patch_size = self.to_list(config.patch_size)

        self.subjects = get_subjects(config)

        self.transforms = self.transform(config)

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue_dataset = Queue(
            self.training_set, queue_length, samples_per_volume, LabelSampler(patch_size=self.patch_size), num_workers=4
        )

    def to_list(self, string):
        return [int(i) for i in string.split(",")]

    def transform(self, config):
        if "train" in config.job_name:
            if "SAM" in config.module_list:
                training_transform = Compose(
                    [
                        # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                        # RandomAffine(degrees=20),
                        # RandomNoise(std=0.0001, exclude=["spatial_source", "intensity_source"]),
                        RandomSwap(
                            patch_size=self.swap_size, num_iterations=config.swap_iterations, include="spatial_source"
                        ),
                        ZNormalization(),
                        # tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
                    ]
                )
            else:
                training_transform = Compose(
                    [
                        # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                        # RandomAffine(degrees=20),
                        # RandomNoise(std=0.0001, exclude=["spatial_source", "intensity_source"]),
                        # RandomSwap(
                        #     patch_size=self.swap_size, num_iterations=config.swap_iterations, include="spatial_source"
                        # ),
                        ZNormalization(),
                        # tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
                    ]
                )
        elif "predict" in config.job_name:
            training_transform = Compose(
                [
                    ZNormalization(),
                ]
            )
        else:
            training_transform = Compose(
                [
                    # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    # RandomAffine(degrees=20),
                    # RandomNoise(std=0.0001),
                    ZNormalization(),
                ]
            )
        return training_transform


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    dataset = SIAMDataLoader(config=config)
    for i in dataset.subjects:
        print(i)


if __name__ == "__main__":
    main()
