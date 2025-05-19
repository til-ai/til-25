"""Manages the ASR model."""

import io
import torch
import librosa
import soundfile as sf
import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration

class ASRManager:

    def __init__(self):
        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(".")
        self.model = WhisperForConditionalGeneration.from_pretrained(".")
        self.model.generation_config.input_ids = self.model.generation_config.forced_decoder_ids
        self.model.generation_config.forced_decoder_ids = None
        # self.model = WhisperForConditionalGeneration.from_pretrained("./")
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
  
    
    def asr(self, audio_bytes: bytes) -> str:
        # Decode WAV from bytes
        with io.BytesIO(audio_bytes) as audio_buffer:
            audio, sr = sf.read(audio_buffer)

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)


        # Whisper expects numpy float32
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features

        if torch.cuda.is_available():
            input_features = input_features.to("cuda")

        # Generate transcription
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return transcription
    
manager = ASRManager()