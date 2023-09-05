# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from pickle import FALSE

import cv2
from mmengine.model.utils import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot

import multiprocessing
from multiprocessing import Process
from itertools import repeat
from threading import Thread

numberOfWorkers = 2
jobsPerWorker = 6

def compute(model, frame, result):
        res = show_result_pyplot(model, frame, result, show=False)
        return res

def process(frames, config, checkpoint): #Worker procedur   
    model = init_model(config, checkpoint ,"cuda:0")
    results = []
    for i in range(jobsPerWorker): # Inference each frame
        results.append(inference_model(model, frames[i]))
        print("worker finsihed inference: " + str(i))
        
        
    output = []
    for i in range(jobsPerWorker): # Combine result with imput frame to get a rendered image
        output.append(compute(model, frames[i], results[i]))
        print("worker finsihed image: " + str(i))
    return output

def processXFrames(frames, numberOfFrames, config, checkpoint):
    model = init_model(config, checkpoint ,"cuda:0")
    print("Starting with process of " + str(numberOfFrames) + " frames!")
    results = []
    for i in range(numberOfFrames): # Inference each frame
        results.append(inference_model(model, frames[i]))        
        
    output = []
    for i in range(numberOfFrames): # Combine result with imput frame to get a rendered image
        output.append(compute(model, frames[i], results[i]))
    print("Finished!")
    return output

def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file or webcam id')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
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

    assert args.show or args.output_file, \
        'At least one output should be enabled.'

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
    if args.output_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
        output_fps = args.output_fps if args.output_fps > 0 else input_fps
        output_height = args.output_height if args.output_height > 0 else int(
            input_height)
        output_width = args.output_width if args.output_width > 0 else int(
            input_width)
        writer = cv2.VideoWriter(args.output_file, fourcc, output_fps,
                                 (output_width, output_height), True)

    

    try: 
        framecount = 0
        videolenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rest = videolenght % (numberOfWorkers * jobsPerWorker)
        videolenght = videolenght - rest

        while framecount < videolenght:
            #Open mutliprocessing pool
            pool = multiprocessing.Pool(processes=numberOfWorkers) 
            #Calculate progress
            framecount += numberOfWorkers * jobsPerWorker
            print("Progress: [" + str(framecount) + "/" + str(videolenght) +"]" ) 
            frames = []
            for _ in range(numberOfWorkers): #for each worker
                workerframes = []
                for _ in range(jobsPerWorker): # create a array of frames to be calcualted
                    ret, frame = cap.read()
                    if ret:
                        workerframes.append(frame)
                frames.append(workerframes)
            print(args.config)
            workerArgs = zip(frames, repeat(args.config), repeat(args.checkpoint))
            outputs = pool.starmap(process, workerArgs)  # Kick-off workers
            pool.close()
            pool.join() # Wait for workes
            print("Joined Pool")
            for item in outputs: # Append inferenced images to video intermimg
                for im in item:
                    draw_img = im
                    if writer:
                        if draw_img.shape[0] != output_height or draw_img.shape[
                                1] != output_width:
                            draw_img = cv2.resize(draw_img,
                                                (output_width, output_height))
                        writer.write(draw_img)
        
        if (rest != 0): #Iterate over the leftover frames (max. frames per worker times number of workers)
            lFrames = []
            for _ in range(rest): 
                    ret, frame = cap.read()
                    if ret:
                        lFrames.append(frame)
            img = processXFrames(lFrames, rest)
            for im in img:
                draw_img = im
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
        pool.close()

if __name__ == '__main__':
    main()
