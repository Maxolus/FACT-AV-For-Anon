# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp


def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file or webcam id')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--output-fourcc',
        default='MJPG',
        type=str,
        help='Fourcc of the output video')
    parser.add_argument(
        '--palette',
        default='ade20k',
        choices=['ade20k', 'cityscapes', 'cocostuff'],
        help='Color palette used for segmentation map')
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
    parser.add_argument(
        '--output-file', default=None, type=str, help='Output video file path')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    

    # build input video
    cap = cv2.VideoCapture(args.video)
    assert (cap.isOpened())
    input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # init output video
    writer = None
    output_height = None
    output_width = None
    if args.output_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
        output_fps = args.output_fps if args.output_fps > 0 else input_fps
        output_height = args.output_height if args.output_height > 0 else int(
            input_height)
        output_width = args.output_width if args.output_width > 0 else int(
            input_width)
        writer = cv2.VideoWriter(args.output_file, fourcc, output_fps,
                                 (output_width, output_height), True)

    # start looping
    try:
        framecount = 0
        videolenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            flag, frame = cap.read()
            if not flag:
                break
            
            print("Progress: [" + str(framecount) + "/" + str(videolenght) +"]" ) 
            framecount += 1

            # inference a single image
            result = inference_segmentor(model, frame)
            draw_img = model.show_result(frame, result,
                        palette=get_palette(args.palette),
                        show=False, opacity=args.opacity)
            
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

    """    
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(args.img, result,
                            palette=get_palette(args.palette),
                            show=False, opacity=args.opacity)
    mmcv.mkdir_or_exist(args.out)
    out_path = osp.join(args.out, osp.basename(args.img))
    cv2.imwrite(out_path, img)
    print(f"Result is save at {out_path}")
"""
if __name__ == '__main__':
    main()