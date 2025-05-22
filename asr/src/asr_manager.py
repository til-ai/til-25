"""Manages the ASR model."""

import io
import torch
import torchaudio
import numpy as np
import noisereduce as nr

from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class ASRManager:

    def __init__(self):
        # Load processor and model
        # Whisper
        self.processor = WhisperProcessor.from_pretrained(".")
        self.model = WhisperForConditionalGeneration.from_pretrained(".")
        # self.model.generation_config.input_ids = self.model.generation_config.forced_decoder_ids
        # self.model.generation_config.forced_decoder_ids = None
        
        # W2V2
        # self.processor = Wav2Vec2Processor.from_pretrained(".")
        # self.model = Wav2Vec2ForCTC.from_pretrained(".")
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
  
    
    def asr(self, audio_bytes: bytes) -> str:
        # Decode WAV from bytes
        with io.BytesIO(audio_bytes) as audio_buffer:
            audio, sr = torchaudio.load(audio_buffer)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = resampler(audio)

        audio = nr.reduce_noise(y=audio, sr=sr) # Both tensor n numpy works as input, output as numpy
        # Whisper expects numpy float32
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        
        # Wav2Vec2
        # input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")
        
        # # Generate transcription with Whisper
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
        # Generate transcription with W2V2
        # Forward pass through the model (no need to use .generate)
        # with torch.no_grad():
        #     logits = self.model(input_features).logits
         
        # Decode the logits to get the predicted ids
#         predicted_ids = torch.argmax(logits, dim=-1)
        
#         # Decode the predicted ids to text
#         transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#         return transcription

