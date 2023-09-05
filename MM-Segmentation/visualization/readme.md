### Usage
To change the visualization of mm-segmentation, the following file must be changed in the corresponding installation of mm-segmentation.:
```
..\mmsegmentation\mmseg\visualization\local_visualizer.py
```
The here presented local_visualizer.py can then decide on the basis of the classes (line 111), which colors (GBR) are assigned to them.


## Hints and Notes

| Index    | CLASS         |
|----------|---------------|
| 0        | road          |
| 1        | sidewalk      |
| 2        | building      |
| 3        | wall          |
| 4        | fence         |
| 5        | pole          |
| 6        | traffic light |
| 7        | traffic sign  |
| 8        | vegetation    |
| 9        | terrain       |
| 10       | sky           |
| 11       | person        |
| 12       | rider         |
| 13       | car           |
| 14       | truck         |
| 15       | bus           |
| 16       | train         |
| 17       | motorcycle    |
| 18       | bicycle       |

This Python script utilizes the classes above. In line 137, the skip_index array is employed to specify the classes that will not be used for segmentation. For example, include a 0 if you wish to exclude road segmentation.

Lines 111 to 151 are my own additions, while the remaining code belongs to the original MMSegmentation codebase.

As demonstrated in line 141, when a label isn't present in the skip index, the script excludes the label for a visualization. Subsequently, the color_seg image receives a specific color wherever seg matches the label. This means if the boolean mask is triggered (seg == label) because the prediction has assigned a class the color will be set for this label (its basicly a double loop for each pixel but beautified in one line). The assigned color for this mask is [0, 230, 230], and it's important to note that the color format is in BGR (Blue, Green, Red), NOT in RGB.

```
if label == 6 or label == 7: # IF label that was detected is a traffic light or a traffic sign
    color_seg[seg == label, :] = [0, 230, 230] #Look at every pixel where the label applies (prediction == label (in this case 6 or 7)) and set it to yellow
```

Once the color_seg matrix is initialized, the image is generated using the following procedure:

```
color_seg = (image * (1 - self.alpha) + mask * self.alpha).astype(np.uint8)
```

In this step, the image is modified using the product of the image and 1 minus the alpha value. The overlay effect is influenced by the transparency (alpha) of the mask. Following this adjustment, the color_seg mask is blended with the image using the formula color_seg * alpha. The term (1 minus the alpha value) ensures that the total contribution of alpha is 1. This approach forms a weighted function: for instance, if you set the alpha value to 0.7, it allocates 30% weight to the original image and 70% weight to the color mask overlaid on top of it. Therefore, the alpha value defines the transparency of the overlaid mask. It's important to note that the alpha value can be specified within the range of 0 to 1 [0, 1].