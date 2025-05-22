import io
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import base64
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr

wav_file_path = "/home/jupyter/advanced/asr/sample_0.wav"

with open(wav_file_path, "rb") as wav_file:
        # Read the binary data from the file
    wav_data = wav_file.read()
        
        # Encode the binary data into base64
    base64_encoded = base64.b64encode(wav_data).decode("utf-8")

audio_bytes = base64.b64decode(base64_encoded)

processor = WhisperProcessor.from_pretrained("/home/jupyter/til-25-hihi/asr/whisper_tiny_finetuned")
model = WhisperForConditionalGeneration.from_pretrained("/home/jupyter/til-25-hihi/asr/whisper_tiny_finetuned")
model.generation_config.input_ids = model.generation_config.forced_decoder_ids
model.generation_config.forced_decoder_ids = None
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

with io.BytesIO(audio_bytes) as audio_buffer:
    waveform, sample_rate = torchaudio.load(audio_buffer)

# Convert to mono if stereo
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Whisper expects 16kHz
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Whisper expects numpy float32
waveform = nr.reduce_noise(y=waveform, sr=sample_rate)
input_features = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features

if torch.cuda.is_available():
    input_features = input_features.to("cuda")

# Generate transcription
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(transcription)