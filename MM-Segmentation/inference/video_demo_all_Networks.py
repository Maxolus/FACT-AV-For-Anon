# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from doctest import debug

import cv2
import re
from mmengine.model.utils import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot

def extract_filename(path):
    match = re.search(r'\\([^\\]+)\.mp4$', path)
    if match:
        return match.group(1)
    return None

def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file or webcam id')
    """
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
   

    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show draw result')
    parser.add_argument(
        '--show-wait-time', default=1, type=int, help='Wait time after imshow')
    parser.add_argument(
        '--output-file', default=None, type=str, help='Output video file path')
    """
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    parser.add_argument(
        '--output-fourcc',
        default='MJPG',
        type=str,
        help='Fourcc of the output video')
    parser.add_argument(
        '--output-fps', default=-1, type=int, help='FPS of the output video')
    parser.add_argument(
        '--output-height',
        default=-1,
        type=int,
        help='Frame height of the output video')
    parser.add_argument(
        '--output-width',
        default=-1,
        type=int,
        help='Frame width of the output video')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    #assert args.show or args.output_file, \
       # 'At least one output should be enabled.'

    filename = extract_filename(args.video)

    configPaths = [       
        "..\\configs\\ocrnet\\ocrnet_hr48_4xb2-160k_cityscapes-512x1024.py",
        "..\\configs\\deeplabv3plus\\deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py",
        "..\\configs\\deeplabv3plus\\deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py",
        "..\\configs\\deeplabv3plus\\deeplabv3plus_r101-d16-mg124_4xb2-40k_cityscapes-512x1024.py",
        "..\\configs\\pspnet\\pspnet_r50b-d8_4xb2-80k_cityscapes-512x1024.py",
        "..\\configs\\upernet\\upernet_r50_4xb2-40k_cityscapes-512x1024.py",
        "..\\configs\\deeplabv3plus\\deeplabv3plus_r18-d8_4xb2-80k_cityscapes-769x769.py",
        "..\\configs\\fcn\\fcn_r101-d8_4xb2-80k_cityscapes-512x1024.py",
        "..\\configs\\pspnet\\pspnet_r50-d32_rsb_4xb2-adamw-80k_cityscapes-512x1024.py",
        "..\\configs\\fcn\\fcn_r50-d8_4xb2-80k_cityscapes-512x1024.py",
        "..\\configs\\fcn\\fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py",
        "..\\configs\\fcn\\fcn_r18-d8_4xb2-80k_cityscapes-512x1024.py",
        "..\\configs\\mobilenet_v2\\mobilenet-v2-d8_pspnet_4xb2-80k_cityscapes-512x1024.py",
        "..\\configs\\mobilenet_v3\\mobilenet-v3-d8_lraspp_4xb4-320k_cityscapes-512x1024.py",
        "..\\configs\\cgnet\\cgnet_fcn_4xb8-60k_cityscapes-512x1024.py",
        "..\\configs\\mobilenet_v3\\mobilenet-v3-d8-scratch_lraspp_4xb4-320k_cityscapes-512x1024.py",
        "..\\configs\\cgnet\\cgnet_fcn_4xb4-60k_cityscapes-680x680.py"
    ]

    checkpointPaths = [ 
        "..\\modelPTHs\\ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth",
        "..\\modelPTHs\\deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth",
        "..\\modelPTHs\\deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth",
        "..\\modelPTHs\\deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes_20200908_005644-cf9ce186.pth",
        "..\\modelPTHs\\pspnet_r50b-d8_512x1024_80k_cityscapes_20201225_094315-6344287a.pth",
        "..\\modelPTHs\\upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth",
        "..\\modelPTHs\\deeplabv3plus_r18-d8_769x769_80k_cityscapes_20201226_083346-f326e06a.pth",
        "..\\modelPTHs\\fcn_r101-d8_512x1024_80k_cityscapes_20200606_113038-3fb937eb.pth",
        "..\\modelPTHs\\pspnet_r50-d32_rsb-pretrain_512x1024_adamw_80k_cityscapes_20220316_141229-dd9c9610.pth",
        "..\\modelPTHs\\fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d.pth",
        "..\\modelPTHs\\fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth",
        "..\\modelPTHs\\fcn_r18-d8_512x1024_80k_cityscapes_20201225_021327-6c50f8b4.pth",
        "..\\modelPTHs\\pspnet_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-19e81d51.pth",
        "..\\modelPTHs\\lraspp_m-v3-d8_512x1024_320k_cityscapes_20201224_220337-cfe8fb07.pth",
        "..\\modelPTHs\\cgnet_512x1024_60k_cityscapes_20201101_110254-124ea03b.pth",
        "..\\modelPTHs\\lraspp_m-v3-d8_scratch_512x1024_320k_cityscapes_20201224_220337-9f29cd72.pth",
        "..\\modelPTHs\\cgnet_680x680_60k_cityscapes_20201101_110253-4c0b2f2d.pth"
    ]

    outputPaths = [ 
        "..\\..\\output\\"+filename+"OCRNet-81_35.avi",
        "..\\..\\output\\"+filename+"DeepLabV3+-80_97.avi",
        "..\\..\\output\\"+filename+"DeepLabV3+-80_09.avi",
        "..\\..\\output\\"+filename+"DeepLabV3+-79_09.avi",
        "..\\..\\output\\"+filename+"PSPNet-78_22.avi",
        "..\\..\\output\\"+filename+"UPerNet-77_10.avi",
        "..\\..\\output\\"+filename+"DeepLabV3+-76_26.avi",
        "..\\..\\output\\"+filename+"FCN-75_13.avi",
        "..\\..\\output\\"+filename+"PSPNet-74_09.avi",
        "..\\..\\output\\"+filename+"FCN-73_61.avi",
        "..\\..\\output\\"+filename+"FCN-72_25.avi",
        "..\\..\\output\\"+filename+"FCN-71_11.avi",
        "..\\..\\output\\"+filename+"MobileNetV2 - PSPNet-70_23.avi",
        "..\\..\\Output\\"+filename+"LRASPP - MobileNetV3-69_54.avi",
        "..\\..\\output\\"+filename+"CGNet-68_27.avi",
        "..\\..\\output\\"+filename+"LRASPP - MobileNetV3-67_87.avi",
        "..\\..\\output\\"+filename+"CGNet-65_63.avi"
    ]
    
    for id, val in enumerate(configPaths):
        configPath = configPaths[id]
        checkpointPath = checkpointPaths[id]
        outputPath = outputPaths[id]

        print(checkpointPath)
        print(configPath)

        # build the model from a config file and a checkpoint file
        model = init_model(configPath, checkpointPath, device=args.device)
        #if args.device == 'cpu':
        #    model = revert_sync_batchnorm(model)

        # build input video
        if args.video.isdigit():
            args.video = int(args.video)
        cap = cv2.VideoCapture(args.video)
        assert (cap.isOpened())
        input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        input_fps = cap.get(cv2.CAP_PROP_FPS)

        # init output video
        writer = None
        output_height = None
        output_width = None
        if outputPath is not None:
            fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
            output_fps = args.output_fps if args.output_fps > 0 else input_fps
            output_height = args.output_height if args.output_height > 0 else int(
                input_height)
            output_width = args.output_width if args.output_width > 0 else int(
                input_width)
            writer = cv2.VideoWriter(outputPath, fourcc, output_fps,
                                    (output_width, output_height), True)

        # start looping
        try:
            framecount = 0
            videolenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while True:
                flag, frame = cap.read()
                if not flag:
                    break
                
                print("Progress Video: ["+ str(id) + "/" + str(len(configPaths)) + "] Frame: [" + str(framecount) + "/" + str(videolenght) +"]" ) 
                framecount += 1

                # test a single image
                result = inference_model(model, frame)

                # blend raw image and prediction
                draw_img = show_result_pyplot(model, frame, result, show=False)

                if writer:
                    if draw_img.shape[0] != output_height or draw_img.shape[
                            1] != output_width:
                        draw_img = cv2.resize(draw_img,
                                            (output_width, output_height))
                    writer.write(draw_img)
        finally:
            if writer:
                writer.release()
            cap.release()


if __name__ == '__main__':
    main()
