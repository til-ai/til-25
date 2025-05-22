from datasets import load_dataset
from datasets import Audio
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import torchaudio.transforms as T

import librosa
import numpy as np
import evaluate
import random
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

dataset = load_dataset("json", data_files="./asr_edited.jsonl", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def stereo_to_mono(example):
    audio_array = example["audio"]["array"]
    # If stereo (2D), convert to mono
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=0)
    example["audio"]["array"] = audio_array
    return example

dataset = dataset.map(stereo_to_mono)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

## Resample
def prepare_train_dataset(batch):
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

def prepare_test_dataset(batch):
    # # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

split_dataset = dataset.train_test_split(test_size=0.01, seed=42)

# Access splits
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]


train_dataset = train_dataset.map(prepare_train_dataset)
test_dataset = test_dataset.map(prepare_test_dataset,remove_columns=test_dataset.column_names)

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
# model.generation_config.language = "english"
# model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
# model.generation_config.use_cache = False
# model.generation_config.generation_max_length=225

@dataclass
class DataCollatorSpeechSeq2SeqWithAugment:
    processor: WhisperProcessor
    decoder_start_token_id: int
    noise_level: float = 0.005 
    
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform) * self.noise_level
        return waveform + noise

    def time_stretch(self, audio, rate=1):
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio, sr, n_steps_range=(-2, 2)):
        n_steps = np.random.uniform(*n_steps_range)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    def augment_audio(self, audio, sr=16000):
        if random.random() < 0.5:
            audio = self.add_noise(audio)
        audio_np = audio.detach().cpu().numpy()
        if random.random() < 0.3:
            random_rate = random.uniform(0.9, 1.1)
            audio_np = self.time_stretch(audio_np,random_rate)
        if random.random() < 0.3:
            audio_np = self.pitch_shift(audio_np, sr)
        audio = torch.from_numpy(audio_np).float().to(audio.device)
        return audio

    def spec_augment(self, mel_spec, time_mask_param=30, freq_mask_param=13, prob=0.5):
        if random.random() > prob:
            return mel_spec  # no masking

        # Apply 2 masks each
        for _ in range(2):
            mel_spec = T.TimeMasking(time_mask_param=time_mask_param)(mel_spec)
            mel_spec = T.FrequencyMasking(freq_mask_param=freq_mask_param)(mel_spec)

        return mel_spec

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # print("Batch keys:", features[0].keys())
        if "input_features" not in features[0]:
            input_features = []
            for feature in features:
                waveform = torch.tensor(feature["audio"]["array"])
                sampling_rate = feature["audio"]["sampling_rate"]

                # Add noise
                noisy_waveform = self.augment_audio(waveform)

                # Re-extract features from noisy waveform
                inputs = self.processor.feature_extractor(
                    noisy_waveform.numpy(), sampling_rate=sampling_rate, return_tensors="pt"
                )
                augmented_inputs = self.spec_augment(inputs.input_features[0])
                input_features.append({"input_features": augmented_inputs})

        else:
            # Evaluation mode — use precomputed features
            input_features = [{"input_features": f["input_features"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
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
        
data_collator = DataCollatorSpeechSeq2SeqWithAugment(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper_tiny_finetuned_model1",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
    logging_dir="./whisper_tiny_logs1",
    learning_rate=5e-5,
    # warmup_steps=1000,
    warmup_ratio=0.1,
    num_train_epochs=20,
    # max_steps=200,
    gradient_checkpointing=True,
    fp16=True,
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=256,  
    save_strategy="steps",
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=10,
    save_total_limit=2, 
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

trainer.train(resume_from_checkpoint=True)
trainer.save_model()

predictions_aft_finetuning = trainer.predict(test_dataset)
wer_aft_finetuning = compute_metrics(predictions_aft_finetuning)

print(f"WER aft fine-tuning: {wer_aft_finetuning['wer']}")
