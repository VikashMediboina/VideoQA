from split_audio_vedioe import split_audio_path
from audio_captios import speech_to_text
from utills import remove_allFiles
from vedioe_captions import process_vedioe
from llama2_inference import lama_inference
# from gradio.mix import Parallel

def process_vedioes(vedioe_path):
    catchi_path="/home/mediboina.v/Vikash/CV/catchi"
    audio_path=catchi_path+'/extracted_audio.wav'
    split_audio_path(vedioe_path,audio_path)
    audio_text= speech_to_text(audio_path)
    vedio_text=process_vedioe(vedioe_path)[0]['summary_text']
    remove_allFiles(catchi_path)
    return audio_text, vedio_text


def process(vedioe_path):
    audio_text, vedio_text=process_vedioes(vedioe_path)
    context= f'''<<SYS>>Please ensure accuracy when providing information from both the video and audio sources, as I may inquire based on their content. If uncertain, kindly respond with "don't know" without offering additional information.\n Audio source: {audio_text}\n Video source: {vedio_text}<<SYS>>'''
    previous_inputs=lama_inference("what does women doing? ",context)
    output_text=chat_bot_text=previous_inputs.split('[/INST]')[-1]
    print(output_text)
    return output_text
# print(process('/home/mediboina.v/Vikash/CV/Data/video10.mp4'))

def chatbot_fn(vedio_text, audio_text, question):
    context= f'''<<SYS>>Please ensure accuracy when providing information from both the video and audio sources, as I may inquire based on their content. If uncertain, kindly respond with "don't know" without offering additional information. Audio source: {audio_text} Video source: {vedio_text}<<SYS>>'''
    print(context)
    previous_inputs= lama_inference(question, context)
    output_text=chat_bot_text=previous_inputs.split('[/INST]')[-1]
    # print(output_text)
    return output_text


vedioe_path='/home/mediboina.v/Vikash/CV/Data/video7045.mp4'
audio_text, vedio_text=process_vedioes(vedioe_path)



# def chatbox_interface(response):
#     # Create a Gradio chatbox component with the response from the video processing
#     chatbox = gr.Chatbox("Video Processing Result:\n" + response)

#     return chatbox

# # Define the Gradio interface
# video_input = gr.inputs.File(label="Upload a video", type="video")
# parallel = Parallel(process_videos, chatbox_interface, inputs=video_input)

# # Launch the Gradio interface
# gr.Interface(fn=parallel, 
#              inputs=video_input,
#              outputs=gr.outputs.Component(type="chatbox", label="Video Processing Result"),
#              live=True).launch()

# Implement the loop to ask for questions
while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    response = chatbot_fn(vedio_text, audio_text, question)
    print("Chatbot Response:", response)