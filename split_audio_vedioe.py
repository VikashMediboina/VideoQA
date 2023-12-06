from moviepy.editor import VideoFileClip



def split_audio_path(audio_path,output_path):

# Load video
    video = VideoFileClip(audio_path)

# Extract audio from the video
    audio = video.audio
    audio.write_audiofile(output_path)

# Load the audio file with pydub (if conversion is needed)
# audio_segment = AudioSegment.from_wav("/home/mediboina.v/Vikash/CV/catchi/extracted_audio.wav")

# # Use speech_recognition to transcribe the audio
# recognizer = sr.Recognizer()
# with sr.AudioFile("/home/mediboina.v/Vikash/CVProject/catchi/extracted_audio.wav") as source:
#     audio_data = recognizer.record(source)
    

