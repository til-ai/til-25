from datasets import load_dataset
from datasets import Audio
# from transformers import WhisperFeatureExtractor
# from transformers import WhisperTokenizer
# from transformers import WhisperProcessor
# from transformers import WhisperForConditionalGeneration
# from transformers import Seq2SeqTrainingArguments
# from transformers import Seq2SeqTrainer

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

import evaluate
import jiwer # For WER calculation and normalisation
import numpy as np
import random
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

dataset = load_dataset("json", data_files="./asr.jsonl", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

MODEL_NAME = "facebook/wav2vec2-large-robust-ft-libri-960h"
# MODEL_NAME =  "openai/whisper-large-v3-turbo"
# feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
# tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="English", task="transcribe")
# processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="English", task="transcribe")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_NAME, language="English", task="transcribe")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, language="English", task="transcribe")

## Resample
def prepare_train_dataset(batch):
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

def prepare_test_dataset(batch):
    # # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    # batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0] # Whisper
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"])["input_values"][0] # Wav2Vec2


    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Access splits
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

train_dataset = train_dataset.map(prepare_train_dataset)
test_dataset = test_dataset.map(prepare_test_dataset,remove_columns=test_dataset.column_names)

## Load model and apply LoRA
# model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
# model.generation_config.language = "english"
# model.generation_config.task = "transcribe"
# model.generation_config.forced_decoder_ids = None
# model.generation_config.use_cache = False
# model.generation_config.generation_max_length=225

# LoRA config
lora_config = LoraConfig(
    r=16,  # Rank of the update matrices.
    lora_alpha=32,  # Alpha scaling.
    target_modules=["q_proj", "v_proj"], # Apply LoRA to query and value projections in attention
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Augmentation Probabilities
REVERB_PROB = 0.3     # Probability to apply reverberation
REVERB_PROFILE = [
    ["reverb", "-w", "50", "75", "100"], # Wet-only reverb with varying room scales
    ["reverb", "50", "50", "75"],       # Different reverb params
]

PITCH_SHIFT_PROB = 0.5
PITCH_SHIFT_SEMITONES_MIN = -2 # Min semitones to shift
PITCH_SHIFT_SEMITONES_MAX = 2  # Max semitones to shift

TEMPO_PERTURB_PROB = 0.5 # Probability to apply tempo perturbation
TEMPO_RATE_MIN = 0.85          # Min rate for time stretching
TEMPO_RATE_MAX = 1.15          # Max rate for time stretching

NOISE_PROB = 0.5 # Probability for Gaussian noise
NOISE_LEVEL = 0.005    # Level of Gaussian noise

@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingAndNoise:
    # processor: WhisperProcessor
    processor: Wav2Vec2Processor
    decoder_start_token_id: int
    
    # Reverb
    reverb_prob: float = 0.0
    reverb_profiles: List[List[str]] = field(default_factory=list)
    
    # Pitch shifting
    pitch_shift_prob: float = 0.0
    pitch_shift_semitones_min: float = 0.0
    pitch_shift_semitones_max: float = 0.0
    
    # Tempo perturbation
    tempo_perturb_prob: float = 0.0
    tempo_rate_min: float = 1.0
    tempo_rate_max: float = 1.0
    
    # Gaussian noise
    noise_prob: float = 0.0
    noise_level: float = 0.005
    
        
    def apply_reverb(self, waveform_tensor: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        if not self.reverb_profiles or random.random() > self.reverb_prob:
            return waveform_tensor
        
        profile = random.choice(self.reverb_profiles)
        try:
            # torchaudio.sox_effects expects a 2D tensor [channels, samples]
            if waveform_tensor.ndim == 1:
                waveform_tensor = waveform_tensor.unsqueeze(0)
            
            reverbed_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform_tensor, sampling_rate, [profile], channels_first=True
            )
            return reverbed_waveform.squeeze(0) # Back to 1D if it was originally
        except Exception as e:
            print(f"Warning: Could not apply reverb: {e}")
            return waveform_tensor.squeeze(0) if waveform_tensor.ndim > 1 else waveform_tensor
    
    def apply_pitch_shift(self, waveform_np: np.ndarray, sampling_rate:int) -> np.ndarray:
        if random.random() < self.pitch_shift_prob:
            n_steps = random.uniform(self.pitch_shift_semitones_min, self.pitch_shift_semitones_max)
            try:
                return librosa.effects.pitch_shift(y=waveform_np, sr=sampling_rate, n_steps=n_steps)
            except Exception as e:
                print(f"Warning: Could not apply pitch shift: {e}")
        return waveform_np
    
    def apply_tempo_perturb(self, waveform_np: np.ndarray) -> np.ndarray:
        if random.random() < self.tempo_perturb_prob:
            rate = random.uniform(self.tempo_rate_min, self.tempo_rate_max)
            try:
                # librosa.effects.time_stretch can be slow for long audio.
                # For very long audio, consider torchaudio.transforms.SpeedPerturbation (if Kaldi available)
                # or sox effects for speed. For typical segment lengths, librosa is fine.
                return librosa.effects.time_stretch(y=waveform_np, rate=rate)
            except Exception as e:
                print(f"Warning: Could not apply tempo perturbation: {e}")
        return waveform_np
  
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform) * self.noise_level
        return waveform + noise

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # print("Batch keys:", features[0].keys())
        if "input_features" not in features[0]:
            input_features = []
            for feature in features:
                waveform_np = feature["audio"]["array"].astype(np.float32)
                sampling_rate = feature["audio"]["sampling_rate"]
                
                # Order of NumPy-based augmentations:
                # 1. Tempo Perturbation
                waveform_np = self.apply_tempo_perturb(waveform_np)
                # 2. Pitch Shifting
                waveform_np = self.apply_pitch_shift(waveform_np, sampling_rate)
                
                # Convert np to tensor for torchaudio/tensor-based augmentations
                waveform_tensor = torch.from_numpy(waveform_np.copy()) # Use .copy() if waveform_np is modified by tensor ops
                
                # Tensor-based augmentations:
                # 3. Apply reverberation
                waveform_tensor = self.apply_reverb(waveform_tensor, sampling_rate)
                # 4. Add Gaussian noise
                waveform_tensor = self.add_noise(waveform_tensor)

                # Re-extract features from noisy waveform
                augmented_audio_np = waveform_tensor.numpy()
                inputs = self.processor.feature_extractor(
                    augmented_audio_np, sampling_rate=sampling_rate, return_tensors="pt"
                )
                input_features.append({"input_features": inputs["input_values"][0]})
        else:
            # Evaluation mode — use precomputed features
            input_features = [{"input_features": f["input_features"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenised label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return {
            "input_features": batch["input_features"],
            "labels": batch["labels"],
        }

data_collator = DataCollatorSpeechSeq2SeqWithPaddingAndNoise(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
    
    # --- Configure which augmentations to use ---
    # Reverb
    reverb_prob=REVERB_PROB,
    reverb_profiles=REVERB_PROFILE,

    # Pitch Shifting
    pitch_shift_prob=PITCH_SHIFT_PROB,
    pitch_shift_semitones_min=PITCH_SHIFT_SEMITONES_MIN,
    pitch_shift_semitones_max=PITCH_SHIFT_SEMITONES_MAX,

    # Tempo Perturbation
    tempo_perturb_prob=TEMPO_PERTURB_PROB,
    tempo_rate_min=TEMPO_RATE_MIN,
    tempo_rate_max=TEMPO_RATE_MAX,

    # Gaussian Noise
    noise_prob=NOISE_PROB,
    noise_level=NOISE_LEVEL
)

metric = evaluate.load("wer")
# Define the JiWER transformation pipeline as specified in wiki
jiwer_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.SubstituteRegexes({r"-": r" "}),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.Strip(),
])

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str_normalised = [jiwer_transform(s) for s in pred_str]
    label_str_normalised = [jiwer_transform(s) for s in label_str]
    
    wer_score_fraction = metric_eval.compute(predictions=pred_str_normalised, references=label_str_normalised)
    
    return {"wer": wer_score_fraction * 100}


# training_args = Seq2SeqTrainingArguments(
training_args = TrainingArguments(
    output_dir="./asr_finetuned_model_lora_augmented",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
    logging_dir="./logs",
    learning_rate=1e-4,
    # warmup_steps=1000,
    warmup_ratio=0.1,
    num_train_epochs=5,
    # max_steps=200,
    gradient_checkpointing=True,
    fp16=True,
    per_device_eval_batch_size=2,
    # predict_with_generate=True,
    # generation_max_length=256,  
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=25,
    save_total_limit=2, 
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False
)

# trainer = Seq2SeqTrainer(
trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

# Start training
print("Starting training...")
trainer.train()

# Save the final LoRA adapters
model.save_pretrained("./asr_finetuned_model_lora_augmented/final_lora_adapters")
# To save the full model if needed (adapters merged)
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained("./asr_finetuned_model_lora_augmented/final_merged_model")
# tokenizer.save_pretrained("./asr_finetuned_model_lora_augmented/final_merged_model")
print("Training complete. LoRA adapters saved.")