### Usage

These scripts can be used for inference. They must be placed in the demo folder. The video_demo.py is used to infer a single video. An example for the call of the script could be: 

Example command: 
```
python .\video_demo.py ..\..\Baseline\baseline_snip.mp4 ..\pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py ..\pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --output-file ..\..\Output\result.avi
```

The video_inference_parallel.py script is a alternative to video_demo.py ensures a parallel execution of the inferring process and can speed up the execution. Please note that the number of workers and the number of jobs per worker must be adjusted to the system (an optimum must be found manually) otherwise this script could also lead to a performace loss.

Example command:
```
python .\video_inference_parallel.py ..\..\Baseline\baseline_snip.mp4 ..\pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py ..\pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --output-file ..\..\Output\result.avi
```

The video_demo_all_Networks.py file takes as its starting point a set of network configurations and, when run, performs the inference for a given video for each network configuration.

Example command:
```
python .\video_demo_all_Networks.py ..\..\input\MyVideo.mp4
```

Note that the corresponding checkpoints must first be downloaded from MM-Segmentation's [website](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/model_zoo.md) and be placed into the appropriate folder (ckpts).

