from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import torchaudio
from utills import remove_file
from vedioe_captions import sumrize
# Check if CUDA is available
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#
# Load the pre-trained Wav2Vec 2.0 model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def speech_to_text(audio_file):
    global model
    model= model.to(device)
    waveform, sample_rate = torchaudio.load(audio_file)

    # Resample the audio file if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Tokenize and predict
    input_values = tokenizer(waveform.squeeze().numpy(), return_tensors="pt").input_values
    input_values = input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted ids to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    # res=sumrize(transcription[0])
    print(transcription[0])
    model=model.to('cpu')
    return transcription[0]

# # Path to your audio file (change this to the path of your file)
# audio_file = "/home/mediboina.v/Vikash/CV/catchi/extracted_audio.wav"

# # Convert speech to text
# text = speech_to_text(audio_file)
# print(text)
