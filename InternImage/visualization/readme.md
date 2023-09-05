### Usage
Since InternImage uses the MM engine and installs it as a Conda package, the code for the visualization process is not in the cloned GitHub project, but in the site package. This can be found in the corresponding environment or in the package installed there. Here for example the location of my file:
```
C:\Users\User\anaconda3\envs\internimage\Lib\site-packages\mmseg\models\segmentors\Base.py
```
The here presented Base.py can then decide on the basis of the classes (line 273), which colors (GBR) are assigned to them.

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

This Python script utilizes the classes above. In line 273, the skip_index array is employed to specify the classes that will not be used for segmentation. For example, include a 0 if you wish to exclude road segmentation.

Lines 273 to 287 are my own additions, while the remaining code belongs to the original InternImage codebase.

As demonstrated in line 276, when a label isn't present in the skip index, the script excludes the label for a visualization. Subsequently, the color_seg image receives a specific color wherever seg matches the label. This means if the boolean mask is triggered (seg == label) because the prediction has assigned a class the color will be set for this label (its basicly a double loop for each pixel but beautified in one line). The assigned color for this mask is [0, 230, 230], and it's important to note that the color format is in BGR (Blue, Green, Red), NOT in RGB.

```
if label == 6 or label == 7: # IF label that was detected is a traffic light or a traffic sign
    color_seg[seg == label, :] = [0, 230, 230] #Look at every pixel where the label applies (prediction == label (in this case 6 or 7)) and set it to yellow
```

Once the color_seg matrix is initialized, the image is generated using the following procedure:

```
img = img * (1 - opacity) + color_seg * opacity
```

In this step, the image is adjusted by the product of the image and 1 - opacity. The overlay effect is influenced by the transparency (opacity) of the mask. After this, color_seg * opacity is applied. The term (1 - opacity) ensures that the total opacity added is 1. This operation creates a weighted function: for instance, if you set the opacity to 0.7, it will 30% weight to the original image and 70% weight to the color mask overlaid on top of it. Therefor the opacity is the trancparency of the overlaid mask. The opacity can be set between 0 and 1 [0,1].



