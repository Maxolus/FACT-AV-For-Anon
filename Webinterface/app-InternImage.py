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
    #Initialize Gradio Interface
    session_var = gr.State([])
    gr.Markdown(
    """
    # FACT-AV!
    The Framework for Calibration of Trust in Automated Vehicles.
    Find models with mIoU from 67 to 81 [here](http://127.0.0.1:7860/)!

    """)
    video = gr.Video()
    dropdown_network = gr.Dropdown(
        [
            "UperNet + InternImage - InternImage-XL / 86,2",
            "SegFormerHead + InternImage - InternImage-L / 85,16",
            "UperNet + InternImage - InternImage-L / 83,68",
            "UperNet + InternImage - InternImage-T / 82,58"
        ], value="UperNet + InternImage - InternImage-XL / 86,2", label="Network", info="Choose your Network / mIoU performance on Cityscapes!"
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

        mIoU = "_86_2"

        if network == "UperNet + InternImage - InternImage-XL / 86,2":
            input_config = "configs/cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py"
            input_checkpoint = "ckpts/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth"
            mIoU = "_86_2"
        elif network == "SegFormerHead + InternImage - InternImage-L / 85,16":
            input_config = "configs/cityscapes/upernet_internimage_l_512x1024_160k_cityscapes.py"
            input_checkpoint = "ckpts/upernet_internimage_l_512x1024_160k_cityscapes.pth"
            mIoU = "_85_16"
        elif network == "UperNet + InternImage - InternImage-L / 83,68":
            input_config = "configs/cityscapes/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.py"
            input_checkpoint = "ckpts/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.pth"
            mIoU = "_83_68"
        elif network == "UperNet + InternImage - InternImage-T / 82,58":
            input_config = "configs/cityscapes/upernet_internimage_t_512x1024_160k_cityscapes.py"
            input_checkpoint = "ckpts/upernet_internimage_t_512x1024_160k_cityscapes.pth"
            mIoU = "_82_58"
        
        
        
        device  = 'cuda:0'
        palette = "cityscapes"
        output_fps = -1
        output_fourcc = 'MJPG'

        sessionname = generate_random_string()
        sessionname = sessionname+mIoU
        session.append(sessionname)

        output_file = sessionname + ".avi"



        # build the model from a config file and a checkpoint file
        
        model = init_segmentor(input_config, checkpoint=None, device=device)
        checkpoint = load_checkpoint(model, input_checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = get_classes(palette)

        

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

                # inference a single image
                result = inference_segmentor(model, frame)
                draw_img = model.show_result(frame, result,
                            palette=get_palette(palette),
                            show=False, opacity=0.5)
                
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
            return [
                session,osp.dirname(osp.abspath(__file__)) + "\\" + output_file, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)            
            ]
    #demo = gr.Interface(inference, 
    #                    gr.Video(), 
    #                    "playable_video",)
    
    
    b1.click(inference, inputs=[video,session_var,dropdown_network[0]], outputs=[session_var, video, slider1, slider2, slider3, slider4, slider5, slider6, surveyMarkdown, textinput, b2])
    b2.click(submit, inputs=[session_var, slider1, slider2, slider3, slider4, slider5, slider6, textinput])
    

demo.queue()    
demo.launch()