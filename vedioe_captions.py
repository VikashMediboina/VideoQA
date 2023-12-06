import os
import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, pipeline
import cv2
from utills import remove_allFiles
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning")

def break_video(input_path, output_folder, frame_interval ):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    
    remove_allFiles(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    current_frame = 0
    video_count = 1

    while current_frame < total_frames:
        frames = []
        for _ in range(frame_interval):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            current_frame += 1

        if frames:
            output_path = os.path.join(output_folder, f"segment_{video_count}.mp4")
            video_count += 1

            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            for frame in frames:
                out.write(frame)

            out.release()

    cap.release()

def caption_generator(videos_folder_path):
    global model
    model=model.to(device)



    # Folder containing videos
    videos_folder = videos_folder_path


    final_sentences = []

    # Iterate through each video in the folder
    for video_file in os.listdir(videos_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(videos_folder, video_file)
            #print(f"Processing video: {video_path}")

            # Load video
            container = av.open(video_path)

            # Extract evenly spaced frames from video
            seg_len = container.streams.video[0].frames
            clip_len = model.config.encoder.num_frames
            indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
            frames = []
            container.seek(0)

            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    frames.append(frame.to_ndarray(format="rgb24"))

            # Generate caption
            gen_kwargs = {
                "min_length": 10,
                "max_length": 20,
                "num_beams": 8,
            }
            pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
            if(np.array(frames).shape[0]==gen_kwargs["num_beams"]):

                tokens = model.generate(pixel_values, **gen_kwargs)
                caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
                if caption not in final_sentences:
                    final_sentences.append(caption)
    model=model.to('cpu')
    return final_sentences

def create_paragraph(sentences):
    paragraph = ' '.join(sentences)
    return paragraph

def process_vedioe(video_input_path):
    frame_interval = 20
    output_video_folder = '/home/mediboina.v/Vikash/CV/catchi'

    break_video(video_input_path, output_video_folder, frame_interval=frame_interval)

    sentences_to_join = caption_generator(output_video_folder)

    joined_paragraph = create_paragraph(sentences_to_join)
    
    res=sumrize(joined_paragraph)
    
    print(res)
    return res
def sumrize(paragraph):
    return summarizer(paragraph, max_length=1000, min_length=100, do_sample=False)
# video_input_path = '/home/mediboina.v/Vikash/CV/Data/video10.mp4'
# print(process_vedioe('/home/mediboina.v/Vikash/CV/Data/video10.mp4'))
