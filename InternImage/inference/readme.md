### Useage

The two scripts inference_all.py and video_demo.py must be placed in the segmentation folder (InternImage/segmentation/) of the GitHub project. Video demo takes a video, a config, a checkpoint (.pth) file and an output-path in order to infer a video. 

Example command: 
```
python video_demo.py inputs/MyVideo.mp4 configs/cityscapes/upernet_internimage_t_512x1024_160k_cityscapes.py ckpts/upernet_internimage_t_512x1024_160k_cityscapes.pth --palette cityscapes --output-file inputs/MyInferedVideo.avi
```

Note that the corresponding checkpoints must first be downloaded from internimage's [website](https://github.com/OpenGVLab/InternImage/tree/master/segmentation/configs/cityscapes) and be placed into the appropriate folder (ckpts).

inference_all.py is just a handy little script to run video_demo.py for some different settings I used in my work.