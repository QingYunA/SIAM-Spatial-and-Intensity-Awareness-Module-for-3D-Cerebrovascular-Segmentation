import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torchio as tio
from torchio.transforms import (
    ZNormalization,
)
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import numpy as np
from utils.conf_base import Default_Conf
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
import hydra
from rich.logging import RichHandler
import logging
from accelerate import Accelerator
import shutil

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ! solve warning


def get_logger(config):
    file_handler = logging.FileHandler(os.path.join(config.hydra_path, f"{config.job_name}.log"))
    rich_handler = RichHandler()

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False
    log.info("Successfully create rich logger")

    return log


def predict(model, config, logger):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    # init progress
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )

    # * load model
    # assert type(conf.ckpt) == str, "You must specify the checkpoint path"
    assert isinstance(config.ckpt, str), "You must specify the checkpoint path"
    logger.info(f"load model from:{os.path.join(config.ckpt, config.latest_checkpoint_file)}")
    ckpt = torch.load(os.path.join(config.ckpt, config.latest_checkpoint_file), map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # * load datasetBs
    from SIAM_dataloader import SIAMDataLoader

    dataset = SIAMDataLoader(config).subjects  # ! notice in predict.py should use Dataset(conf).subjects
    znorm = ZNormalization()

    precision_ls, recall_ls, dice_ls, hs95_ls = [], [], [], []

    file_tqdm = progress.add_task("[red]Predicting file", total=len(dataset))

    # *  accelerator prepare
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    # start progess
    progress.start()
    patch_size = [int(i) for i in config.patch_size.split(",")]
    for i, item in enumerate(dataset):
        item = znorm(item)
        grid_sampler = tio.inference.GridSampler(item, patch_size=(patch_size), patch_overlap=(4, 4, 36))
        affine = item["source"]["affine"]
        spacing = item.spacing
        # * dist sampler
        # dist_sampler = torch.utils.data.distributed.DistributedSampler(grid_sampler, shuffle=True)

        # assert conf.batch_size == 1, 'batch_size must be 1 for inference'

        patch_loader = torch.utils.data.DataLoader(
            grid_sampler, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        patch_loader = accelerator.prepare(patch_loader)
        if i == 0:
            batch_tqdm = progress.add_task("[blue]file batch", total=len(patch_loader))
        else:
            progress.reset(batch_tqdm, total=len(patch_loader))

        pred_aggregator = tio.inference.GridAggregator(grid_sampler)
        gt_aggregator = tio.inference.GridAggregator(grid_sampler)
        with torch.no_grad():
            for j, batch in enumerate(patch_loader):
                locations = batch[tio.LOCATION]

                x = batch["source"]["data"]
                gt = batch["gt"]["data"]

                x = x.type(torch.FloatTensor).to(accelerator.device)
                gt = gt.type(torch.FloatTensor).to(accelerator.device)

                pred = model(x, x, x)

                mask = torch.sigmoid(pred.clone())
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                # mask = pred.clone()
                # mask = mask.argmax(dim=1, keepdim=True)

                pred_aggregator.add_batch(mask, locations)
                gt_aggregator.add_batch(gt, locations)
                progress.update(batch_tqdm, completed=j + 1)
                progress.refresh()
            # reset batchtqdm
            pred_t = pred_aggregator.get_output_tensor()
            gt_t = gt_aggregator.get_output_tensor()

            # * save pred mhd file
            save_mhd(pred_t, affine, i, config)

            # * calculate metrics
            precision, recall, dice, hs95 = metric(gt_t, pred_t, spacing)
            precision_ls.append(precision)
            recall_ls.append(recall)
            dice_ls.append(dice)
            hs95_ls.append(hs95)
            logger.info(
                f"File {i+1} metrics: " f"\nprecision: {precision}" f"\nrecall: {recall}" f"\ndice: {dice}" f"\nhs95: {hs95}"
            )
        progress.update(file_tqdm, completed=i + 1)

    save_csv(
        save_path=config.hydra_path,
        precision_ls=precision_ls,
        recall_ls=recall_ls,
        dice_ls=dice_ls,
        hs95_ls=hs95_ls,
    )
    precision_mean = np.mean(precision_ls)
    recall_mean = np.mean(recall_ls)
    dice_mean = np.mean(dice_ls)
    hs95_mean = np.mean(hs95_ls)
    logger.info(
        f"\nprecision_mean: {precision_mean}"
        f"\nrecall_mean: {recall_mean}"
        f"\ndice_mean: {dice_mean}"
        f"\nhs95_mean: {hs95_mean}"
    )


# def save_csv(jaccard_ls, dice_ls, config):
#     import pandas as pd
#
#     data = {"jaccard": jaccard_ls, "dice": dice_ls}
#     df = pd.DataFrame(data)
#     df.loc[len(df)] = [df.iloc[:, 0].mean(), df.iloc[:, 1].mean()]
#     save_path = os.path.join(config.hydra_path, "metrics.csv")
#     df.to_csv(save_path, index=False)


def save_csv(save_path, **kwargs):
    import pandas as pd

    data = {}
    for key, value in kwargs.items():
        data[key] = value

    df = pd.DataFrame(data)
    df.loc[len(df)] = [df.iloc[:, i].mean() for i in range(df.shape[1])]
    save_path = os.path.join(save_path, "metrics.csv")
    df.to_csv(save_path, index=True)


def save_mhd(pred, affine, index, config):
    save_base = os.path.join(config.hydra_path, "pred_file")
    os.makedirs(save_base, exist_ok=True)
    pred_data = tio.ScalarImage(tensor=pred, affine=affine)
    pred_data.save(os.path.join(save_base, f"pred-{index:04d}.mhd"))


@hydra.main(config_path="conf", config_name="config")
def main(config):
    config = config["config"]

    os.makedirs(config.hydra_path, exist_ok=True)
    if config.network == "res_unet":
        from models.three_d.residual_unet3d import UNet

        model = UNet(in_channels=config.in_classes, n_classes=config.out_classes, base_n_filter=32)
    elif config.network == "unet":
        from models.three_d.unet3d import UNet3D  # * 3d unet

        model = UNet3D(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == "er_net":
        from models.three_d.ER_net import ER_Net

        model = ER_Net(classes=config.out_classes, channels=config.in_classes)
    elif config.network == "SIAMUNet":
        from models.three_d.SIAM_Unet import SIAMUNet

        model = SIAMUNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)

    # * create logger
    logger = get_logger(config)
    info = "\nParameter Settings:\n"
    for k, v in config.items():
        info += f"{k}: {v}\n"
    logger.info(info)

    predict(model, config, logger)


if __name__ == "__main__":
    main()
