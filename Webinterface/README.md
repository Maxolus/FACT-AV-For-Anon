



# Web Interface

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

