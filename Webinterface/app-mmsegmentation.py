from argparse import ArgumentParser
from doctest import debug
from turtle import hideturtle

import cv2
from mmengine.model.utils import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot

import os.path as osp
import gradio as gr
import random
import string
import csv

def generate_random_string(length=16):
    # Define the characters that can be used in the random string
    characters = string.ascii_letters + string.digits

    # Generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string

def hideSurvey():
    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]

def extract_highest_element(input_list):
    if not input_list:
        return None  # Return None if the list is empty

    highest_element = max(input_list, key=lambda x: x)
    return highest_element

def submit(session, slider1, slider2, slider3, slider4, slider5, slider6, textinput):
    # Split the session name
    sessionname = extract_highest_element(session)


    # Define the CSV file name based on the session name
    csv_filename = f"{sessionname}.csv"

    # Split the string at the first '_'
    parts = sessionname.split('_', 1)

    session_name = parts[0]
    mIoU = parts[1].replace('_', '.')

     # Create a list containing the session name and slider values
    data = [session_name, mIoU, slider1, slider2, slider3, slider4, slider5, slider6, textinput]

    # Write the data to the CSV file
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["SessionName", "mIoU", "Slider1", "Slider2", "Slider3", "Slider4", "Slider5", "Slider6", "Textinput"])
        csv_writer.writerow(data)

    gr.Info("Thank you for yor submission!")


demo = gr.Blocks(title="FACT-AV")

with demo:
        
    session_var = gr.State([])
    gr.Markdown(
    """
    # FACT-AV!
    The Framework for Calibration of Trust in Automated Vehicles.
    Find models with mIoU from 82 to 86 [here](http://127.0.0.1:7861/)!
    
    """)
    video = gr.Video()
    dropdown_network = gr.Dropdown(
        [
            "OCRNet / 81,35",
            "DeepLabV3+ / 80,97",
            "DeepLabV3+ / 80,09",
            "DeepLabV3+ / 79,09",
            "PSPNet / 78,22",
            "UPerNet / 77,1",
            "DeepLabV3+ / 76,26",
            "FCN / 75,13",
            "PSPNet / 74,09",
            "FCN / 73,61",
            "FCN / 72,25",
            "FCN / 71,11",
            "MobileNetV2 - PSPNet / 70,23",
            "LRASPP - MobileNetV3 / 69,54",
            "CGNet / 68,27",
            "LRASPP - MobileNetV3 / 67,87"
        ], value="OCRNet / 81,35", label="Network", info="Choose your Network / mIoU performance on Cityscapes!"
    ),

    

    b1 = gr.Button("Segment")

    surveyMarkdown = gr.Markdown(
    """
    # Survey!
    It would help us, please only aswer if you have withnessed the video.
    PLEASE! only submit if you have generated and watched the video!
    """, visible=False)
    slider1 = gr.Slider(label="How much mental demand was involved in receiving and processing information (e.g., thinking, deciding, calculating, remembering, looking, searching ...)? Was the task easy or challenging, simple or complex, does it require high accuracy or is it error tolerant?", value=10, minimum=1, maximum=20, step=1, visible=False)
    slider2 = gr.Slider(label="I felt safe during the ride.", value=2, minimum=1, maximum=7, step=1, visible=False)
    slider3 = gr.Slider(label="I trust the highly automated vehicle.", value=2, minimum=1, maximum=5, step=1, visible=False)
    slider4 = gr.Slider(label="I was able to understand why things happened.", value=2, minimum=1, maximum=5, step=1, visible=False)
    slider5 = gr.Slider(label="How would you rate the driving style of the automated vehicle? (1 = completely safe, 7 = completely dangerous)", value=3, minimum=1, maximum=7, step=1, visible=False)
    slider6 = gr.Slider(label="I think the visualizations provided were reasonable.", value=2, minimum=1, maximum=7, step=1, visible=False)
    textinput = gr.Textbox(label="Feedback", placeholder="Please feel free to enter Feedback here.", visible=False)
                        
    b2 = gr.Button("Submit Survey", visible=False)

    def inference(video, session, network):

        input_checkpoint = "ckpts/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth"
        input_config = "configs/cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py"

        mIoU = "_81_35"

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
        ]
        

        checkpointPaths = [ 
            "..\\ckpts\\ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth",
            "..\\ckpts\\deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth",
            "..\\ckpts\\deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth",
            "..\\ckpts\\deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes_20200908_005644-cf9ce186.pth",
            "..\\ckpts\\pspnet_r50b-d8_512x1024_80k_cityscapes_20201225_094315-6344287a.pth",
            "..\\ckpts\\upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth",
            "..\\ckpts\\deeplabv3plus_r18-d8_769x769_80k_cityscapes_20201226_083346-f326e06a.pth",
            "..\\ckpts\\fcn_r101-d8_512x1024_80k_cityscapes_20200606_113038-3fb937eb.pth",
            "..\\ckpts\\pspnet_r50-d32_rsb-pretrain_512x1024_adamw_80k_cityscapes_20220316_141229-dd9c9610.pth",
            "..\\ckpts\\fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d.pth",
            "..\\ckpts\\fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth",
            "..\\ckpts\\fcn_r18-d8_512x1024_80k_cityscapes_20201225_021327-6c50f8b4.pth",
            "..\\ckpts\\pspnet_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-19e81d51.pth",
            "..\\ckpts\\lraspp_m-v3-d8_512x1024_320k_cityscapes_20201224_220337-cfe8fb07.pth",
            "..\\ckpts\\cgnet_512x1024_60k_cityscapes_20201101_110254-124ea03b.pth",
            "..\\ckpts\\lraspp_m-v3-d8_scratch_512x1024_320k_cityscapes_20201224_220337-9f29cd72.pth",
        ]

        if network == "OCRNet / 81,35":
            mIoU = "_81_35"
            input_config = configPaths[0]
            input_checkpoint = checkpointPaths[0]
        elif network == "DeepLabV3+ / 80,97":
            mIoU = "_80_97"
            input_config = configPaths[1]
            input_checkpoint = checkpointPaths[1]
        elif network == "DeepLabV3+ / 80,09":
            mIoU = "_80_09"
            input_config = configPaths[2]
            input_checkpoint = checkpointPaths[2]
        elif network == "DeepLabV3+ / 79,09":
            mIoU = "_79_09"
            input_config = configPaths[3]
            input_checkpoint = checkpointPaths[3]
        elif network == "PSPNet / 78,22":
            mIoU = "_78_22"
            input_config = configPaths[4]
            input_checkpoint = checkpointPaths[4]
        elif network == "UPerNet / 77,1":
            mIoU = "_77_1"
            input_config = configPaths[5]
            input_checkpoint = checkpointPaths[5]
        elif network == "DeepLabV3+ / 76,26":
            mIoU = "_76_26"
            input_config = configPaths[6]
            input_checkpoint = checkpointPaths[6]
        elif network == "FCN / 75,13":
            mIoU = "_75_13"
            input_config = configPaths[7]
            input_checkpoint = checkpointPaths[7]
        elif network == "PSPNet / 74,09":
            mIoU = "_74_09"
            input_config = configPaths[8]
            input_checkpoint = checkpointPaths[8]
        elif network == "FCN / 73,61":
            mIoU = "_73_61"
            input_config = configPaths[9]
            input_checkpoint = checkpointPaths[9]
        elif network == "FCN / 72,25":
            mIoU = "_72_25"
            input_config = configPaths[10]
            input_checkpoint = checkpointPaths[10]
        elif network == "FCN / 71,11":
            mIoU = "_71_11"
            input_config = configPaths[11]
            input_checkpoint = checkpointPaths[11]
        elif network == "MobileNetV2 - PSPNet / 70,23":
            mIoU = "_70_23"
            input_config = configPaths[12]
            input_checkpoint = checkpointPaths[12]
        elif network == "LRASPP - MobileNetV3 / 69,54":
            mIoU = "_69_54"
            input_config = configPaths[13]
            input_checkpoint = checkpointPaths[13]
        elif network == "CGNet / 68,27":
            mIoU = "_68_27"
            input_config = configPaths[14]
            input_checkpoint = checkpointPaths[14]
        elif network == "LRASPP - MobileNetV3 / 67,87":
            mIoU = "_67_87"
            input_config = configPaths[15]
            input_checkpoint = checkpointPaths[15]
     
        device  = 'cuda:0'
        output_fps = -1
        output_fourcc = 'MJPG'

        sessionname = generate_random_string()
        sessionname = sessionname+mIoU
        session.append(sessionname)

        output_file = sessionname + ".avi"

        
         # build the model from a config file and a checkpoint file
        model = init_model(input_config, input_checkpoint, device=device)
        

        # build input video
        cap = cv2.VideoCapture(video)
        assert (cap.isOpened())
        input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        input_fps = cap.get(cv2.CAP_PROP_FPS)

        # init output video
        writer = None
        if output_file is not None:
            fourcc = cv2.VideoWriter_fourcc(*output_fourcc)
            output_fps = input_fps
            output_height = int(input_height)
            output_width = int(input_width)
            print(f"Video: {video}")
            print(f"Output File: {output_file}")
            print(f"FourCC Codec: {fourcc}")
            print(f"Output FPS: {output_fps}")
            print(f"Output Resolution: {output_width}x{output_height}")
            print(f"Is Color Video: True")
            print("Outputfile: " + osp.dirname(video) + "\\" + output_file)
        
            writer = cv2.VideoWriter(output_file, fourcc, output_fps,
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
            filepath = osp.dirname(osp.abspath(__file__)) + "\\" + output_file
            return [session, filepath, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]
    #demo = gr.Interface(inference, 
    #                    gr.Video(), 
    #                    "playable_video",)
    

    b1.click(inference, inputs=[video,session_var,dropdown_network[0]], outputs=[session_var, video, slider1, slider2, slider3, slider4, slider5, slider6, surveyMarkdown, textinput, b2])
    b2.click(submit, inputs=[session_var, slider1, slider2, slider3, slider4, slider5, slider6, textinput])
    

demo.queue()    
demo.launch()