with io.BytesIO(audio_bytes) as audio_buffer:
            audio, sr = sf.read(audio_buffer)

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        audio = nr.reduce_noise(y=audio, sr=sr)
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