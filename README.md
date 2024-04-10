# SIAM: Spatial and Intensity Awareness Module for 3D Cerebrovascular Segmentation
This repository is the official implementation of the paper "SIAM: Spatial and Intensity Awareness Module for 3D Cerebrovascular Segmentation," submitted to ACM MM 2024.

## Usage

### Requirements

The recommend python and package version:

* python 3.10.0
* pytorch 1.13.1 (do not use torch2.0, it will cause program failed to start)

### Train

here we use an example(Traning 3D Unet) to teach you how use this repository

```BASH
python train.py config=unet
```
To specify the folder name for the current save, you can modify the corresponding parameter using config.XXX=XXX.
```BASH
python train.py config=unet config.name=unet-1
```
All files during the model training process will be saved in ./logs/unet-1

all parameter can be setted in `conf/unet.yaml`
#### Global configuration
Considering that many settings are common to all configuration files, such as `data_path`, `num_epochs`, etc., to avoid repetitive work, we have placed these common parameters in `conf/config.yaml`. All configuration files will have these properties from `config.yaml`.

Taking `num_workers` (defaulted to 18 in `config.yaml`) as an example, the priority of parameter overriding is as follows:
Command line argument `config.num_workers=20` > `num_workers=18` in `data_3d.yaml` > Default value `num_workers=18` in `config.yaml`.

#### File Structure
Traning logs will be saved like this:
```
./logs/ying_tof (Corresponding saved folder: ./logs/config.name)
└── 2023-11-24 (Date: year-month-day)
    └── 17-05-02 (Time: hour-minute-second)
        ├── .hydra (Configuration save files)
        │   ├── config.yaml (Configuration for this running)
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── train_3d.log (Log during training)
        └── train_tensorboard (Tensorboard folder)
            └── events.out.tfevents.1700816703.medai.115245.0 (file for Tensorboard)
```
### Predict

run the code

```BASH
python predict.py config=unet config.ckpt=XXX
```
`WARNING`: ckpt must be the absolute path of the model, not the relative path
```
./results/ (Root folder for results)
└── unet (Model name: unet)
    └── 2023-12-04 (Date: year-month-day)
        └── 17-39-30 (Time: hour-minute-second)
            ├── metrics.csv (CSV file containing metrics)
            ├── pred_file (Folder for prediction files)
            │   ├── pred-0000.mhd (Prediction file 0 in MHD format)
            │   ├── pred-0000.zraw (Prediction file 0 in ZRAW format)
            │   ├── pred-0001.mhd (Prediction file 1 in MHD format)
            │   ├── pred-0001.zraw (Prediction file 1 in ZRAW format)
            │   ├── pred-0002.mhd (Prediction file 2 in MHD format)
            │   └── ... (Additional prediction files)
            └── predict.log (Log file for prediction)
```


## Create your own configuration

create new file in path `/conf/config`, file name ends with `.yaml`.

For example, if you wanna use 3D Vnet, you can create `vnet.yaml`, and then set all parameters you wanna set.

### how to use the parameters in `yaml` file

`config.PARAM`: replace `PARAM` with the parameters you wanna use

### Modify train.py and predict.py to match your model

in train.py, add these codes after the last model

```Python
elif config.network == 'NETWORK':
    from models.three_d.NETWORK import NETWORK
    model = NETWORK()
```

`NETWORK`: means the network you wanna use
![](https://s2.loli.net/2023/10/26/LEQt8p7TufXxqyb.png)
