



# Web Interface
------------------------------------------------------------------------

# 🔧 Installation Guide (Verified Working Setup for Windows + GPU)

This installation guide is based on extensive debugging to ensure a
**stable, GPU-accelerated**, and **Windows-compatible** environment
without the common MMCV/MMEngine issues such as:

-   `ModuleNotFoundError: mmcv._ext`
-   NumPy 2.x ABI errors
-   mmcv/mmseg version conflicts

Follow the steps exactly for a reliable installation.

------------------------------------------------------------------------

## 1️⃣ Create a Clean Conda Environment

``` bash
conda create -n factav python=3.10 -y
conda activate factav
```

------------------------------------------------------------------------

## 2️⃣ Install Compatible NumPy + PyTorch (CUDA 11.8)

``` bash
pip install numpy==1.26.4
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2     --index-url https://download.pytorch.org/whl/cu118
```

Verify:

``` bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

------------------------------------------------------------------------

## 3️⃣ Install OpenMMLab Libraries (IMPORTANT: use openmim)

``` bash
pip install -U openmim
mim install "mmengine>=0.7.4,<1.0.0"
mim install "mmcv==2.1.0"
mim install "mmsegmentation==1.2.2"
```

Verify MMCV CUDA extension:

``` bash
python -c "import mmcv, mmcv._ext; print('MMCV extension loaded correctly')"
```

------------------------------------------------------------------------

## 4️⃣ Install Project Dependencies

``` bash
pip install gradio==6.0.2 pandas==2.3.3 ftfy==6.2.3
```

------------------------------------------------------------------------

## Requirements

- Ensure you have installed the MMSegmentation and Internimage Framework in a Conda environment.
- FFMPEG should be installed.
- Download all necessary models, or at least the models you intend to use.

## Setting up the Gradio Web Interface

To use the Gradio Web Interface, first install Gradio in each environment:

```bash
pip install gradio
```

Then, place the files from this folder into their corresponding directories:

- `app-mmsegmentation.py` should go into `\mmsegmentation\demo`.
- `app-InternImage.py` should be placed in `InternImage\segmentation`.

These scripts utilize pretrained models from folders named `ckpts`. Create these folders and insert the downloaded models (refer to the table in the readme) from the MMSegmentation and InternImage websites.

- For InternImage, place the models in `InternImage\segmentation\ckpts\[...insert Models]`.
- For MMSegmentation, the models go in `\mmsegmentation\ckpts\[...insert Models]`.

After completing these steps, you should be able to run the apps with Python in their respective environments.

