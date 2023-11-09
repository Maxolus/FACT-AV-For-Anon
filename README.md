



# Running FACT-AV locally



## Getting started

First, the respective projects MM-Segmentation and InternImage must be installed in a Conda environment using the tutorials presented by the developers. Both projects have installation instructions in their documentation, which may differ depending on the system configuration, for this reason, the installation is not described further here.

Additionally, you will need to download the pretrained models listed in the table below from the official MMSegmentation and InternImage Model Zoos in order to use them. These models must then be placed in a folder of your choice; you may need to adjust the paths to the models in the code accordingly.

## Using this Project

This project includes all the scripts we created for the inference process, plus scripts to change the visualizers to match the color scheme used for our work.

The two subfolders contain on the one hand a infrence folder, in this infrence folder all inference scripts are contained which are described shortly in the corresponding readme. The visualization folder contains the modified scripts for the visualization.


## Webinterface

We also offer a web interface implemented with Gradio in order to use FACT-AV. Therefore, you need to place the corresponding app.py file in the project's directory (MMSegmentation or InternImage).

The app.py will run a local web server hosting a site where you can drag and drop a video in order to receive a segmented video in return. Note that you'll have to execute the app.py within the corresponding Conda environment.

# Used Models and mIoU



| Name                                        | mIoU  | Config                                                          | Model                                                                                  | Command LaTex                 |
|---------------------------------------------|-------|-----------------------------------------------------------------|----------------------------------------------------------------------------------------|-------------------------------|
| UperNet + InternImage - InternImage-XL      | 86.2  | upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py    | upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth                          | \cite{wang2023internimage}    |
| SegFormerHead + InternImage - InternImage-L | 85.16 | segformer_internimage_l_512x1024_160k_mapillary2cityscapes.py   | segformer_internimage_l_512x1024_160k_mapillary2cityscapes.pth                         | \cite{wang2023internimage}    |
| UperNet + InternImage - InternImage-L       | 83.68 | upernet_internimage_l_512x1024_160k_cityscapes.py               | upernet_internimage_l_512x1024_160k_cityscapes.pth                                     | \cite{wang2023internimage}    |
| UperNet + InternImage - InternImage-T       | 82.58 | upernet_internimage_t_512x1024_160k_cityscapes.py               | upernet_internimage_t_512x1024_160k_cityscapes.pth                                     | \cite{wang2023internimage}    |
| OCRNet                                      | 81.35 | ocrnet_hr48_4xb2-160k_cityscapes-512x1024.py                    | ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth                      | \cite{yuan2021segmentation}   |
| DeepLabV3+                                  | 80.97 | deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py           | deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth             | \cite{chen2018encoder}        |
| DeepLabV3+                                  | 80.09 | deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py            | deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth              | \cite{chen2018encoder}        |
| DeepLabV3+                                  | 79.09 | deeplabv3plus_r101-d16-mg124_4xb2-40k_cityscapes-512x1024.py    | deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes_20200908_005644-cf9ce186.pth      | \cite{chen2018encoder}        |
| PSPNet                                      | 78.22 | pspnet_r50b-d8_4xb2-80k_cityscapes-512x1024.py                  | pspnet_r50b-d8_512x1024_80k_cityscapes_20201225_094315-6344287a.pth                    | \cite{zhao2017pyramid}        |
| UPerNet                                     | 77.1  | upernet_r50_4xb2-40k_cityscapes-512x1024.py                     | upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth                       | \cite{xiao2018unified}        |
| DeepLabV3+                                  | 76.26 | deeplabv3plus_r18-d8_4xb2-80k_cityscapes-769x769.py             | deeplabv3plus_r18-d8_769x769_80k_cityscapes_20201226_083346-f326e06a.pth               | \cite{chen2018encoder}        |
| FCN                                         | 75.13 | fcn_r101-d8_4xb2-80k_cityscapes-512x1024.py                     | fcn_r101-d8_512x1024_80k_cityscapes_20200606_113038-3fb937eb.pth                       | \cite{long2015fully}          |
| PSPNet                                      | 74.09 | pspnet_r50-d32_rsb_4xb2-adamw-80k_cityscapes-512x1024.py        | pspnet_r50-d32_rsb-pretrain_512x1024_adamw_80k_cityscapes_20220316_141229-dd9c9610.pth | \cite{zhao2017pyramid}        |
| FCN                                         | 73.61 | fcn_r50-d8_4xb2-80k_cityscapes-512x1024.py                      | fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d.pth                        | \cite{long2015fully}          |
| FCN                                         | 72.25 | fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py                      | fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth                        | \cite{long2015fully}          |
| FCN                                         | 71.11 | fcn_r18-d8_4xb2-80k_cityscapes-512x1024.py                      | fcn_r18-d8_512x1024_80k_cityscapes_20201225_021327-6c50f8b4.pth                        | \cite{long2015fully}          |
| MobileNetV2 - PSPNet                        | 70.23 | mobilenet-v2-d8_pspnet_4xb2-80k_cityscapes-512x1024.py          | pspnet_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-19e81d51.pth                    | \cite{sandler2018mobilenetv2} |
| LRASPP - MobileNetV3                        | 69.54 | mobilenet-v3-d8_lraspp_4xb4-320k_cityscapes-512x1024.py         | lraspp_m-v3-d8_512x1024_320k_cityscapes_20201224_220337-cfe8fb07.pth                   | \cite{howard2019searching}    |
| CGNet                                       | 68.27 | cgnet_fcn_4xb8-60k_cityscapes-512x1024.py                       | cgnet_512x1024_60k_cityscapes_20201101_110254-124ea03b.pth                             | \cite{wu2020cgnet}            |
| LRASPP - MobileNetV3                        | 67.87 | mobilenet-v3-d8-scratch_lraspp_4xb4-320k_cityscapes-512x1024.py | lraspp_m-v3-d8_scratch_512x1024_320k_cityscapes_20201224_220337-9f29cd72.pth           | \cite{howard2019searching}    |


