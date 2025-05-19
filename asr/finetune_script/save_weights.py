from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

# Load the processor and model (This downloads the necessary files)
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")

# Save them to a local cache directory
processor.save_pretrained("/home/jupyter/til-25-hihi/asr/whisper-medium.en")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")
model.save_pretrained("/home/jupyter/til-25-hihi/asr/whisper-medium.en")