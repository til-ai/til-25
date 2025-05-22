# import torch

# # Simulate stereo audio: shape (2, num_samples)
# stereo_audio = torch.randn(2, 16000)  # 1 second of stereo audio at 16kHz
# sample_rate = 16000

# # Run logic
# if stereo_audio.shape[0] > 1:
#     stereo_audio = stereo_audio.mean(dim=0, keepdim=True)

# print("Shape after converting to mono:", stereo_audio.shape)

# import torchaudio

# # Simulate mono audio at 8kHz
# mono_audio = torch.randn(1, 8000)  # 1 second at 8kHz
# sample_rate = 8000

# # Resample if needed
# if sample_rate != 16000:
#     resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#     mono_audio = resampler(mono_audio)

# print("Shape after resampling to 16kHz:", mono_audio.shape)  # Should be (1, 16000)

import torch
import numpy as np
import noisereduce as nr

# Sample rate
sr = 16000

# Create synthetic audio: 1 second of noise
tensor_audio = torch.randn(1, sr)   # Shape: (1, 16000) PyTorch tensor
numpy_audio = tensor_audio.squeeze().numpy()  # Convert to 1D NumPy array

# --- Test 1: Pass PyTorch tensor directly ---
try:
    print(type(tensor_audio))
    print("\n[TEST 1] Passing PyTorch tensor directly:")
    reduced_tensor = nr.reduce_noise(y=tensor_audio, sr=sr)
    print("✅ Succeeded (unexpected!) - Returned type:", type(reduced_tensor))
except Exception as e:
    print("❌ Failed as expected -", str(e))

# --- Test 2: Pass NumPy array ---
try:
    print(type(numpy_audio))
    print("\n[TEST 2] Passing NumPy array:")
    reduced_numpy = nr.reduce_noise(y=numpy_audio, sr=sr)
    print("✅ Succeeded - Returned type:", type(reduced_numpy))
except Exception as e:
    print("❌ Failed -", str(e))
