import io
import torch
# import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import base64
import librosa
import soundfile as sf
import numpy as np

wav_file_path = "/home/jupyter/advanced/asr/sample_0.wav"

with open(wav_file_path, "rb") as wav_file:
        # Read the binary data from the file
    wav_data = wav_file.read()
        
        # Encode the binary data into base64
    base64_encoded = base64.b64encode(wav_data).decode("utf-8")

audio_bytes = base64.b64decode(base64_encoded)

processor = WhisperProcessor.from_pretrained("/home/jupyter/til-25-hihi/asr/whisper-small.en")
model = WhisperForConditionalGeneration.from_pretrained("/home/jupyter/til-25-hihi/asr/whisper-small.en")
model.generation_config.input_ids = model.generation_config.forced_decoder_ids
model.generation_config.forced_decoder_ids = None
model.eval()
if torch.cuda.is_available():
    model.to("cuda")
    
with io.BytesIO(audio_bytes) as audio_buffer:
    audio, sr = sf.read(audio_buffer)

if audio.ndim == 2:
    audio = np.mean(audio, axis=1)
    
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# # Whisper expects 16kHz
# if sample_rate != 16000:
#     resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#     waveform = resampler(waveform)

# # Convert to mono if stereo
# if waveform.shape[0] > 1:
#     waveform = waveform.mean(dim=0, keepdim=True)

# Whisper expects numpy float32
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

if torch.cuda.is_available():
    input_features = input_features.to("cuda")

# Generate transcription
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(transcription)