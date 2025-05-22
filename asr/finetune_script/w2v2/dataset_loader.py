import re
import soundfile as sf
import pandas as pd

from datasets import Audio
from datasets import load_dataset


# data_dir=./asr.jsonl

def custom_load_dataset(data_dir,test_size=0.05, seed=42):
    dataset = load_dataset("json", data_files=data_dir, split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)

    # Access splits
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    return train_dataset, test_dataset

## Resample
# def prepare_train_dataset(processor, batch):
#     # encode target text to label ids 
#     batch["labels"] = processor(text=batch["transcript"]).input_ids
#     return batch

def prepare_dataset(processor, batch):
    # # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array 
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    # encode target text to label ids 
    batch["labels"] = processor(text=batch["transcript"]).input_ids
    
    return batch

def custom_load_prepared_dataset(data_dir, processor, test_size=0.05, seed=42):
    train_dataset, test_dataset = custom_load_dataset(data_dir, test_size=test_size, seed=seed)

    train_dataset = train_dataset.map(lambda batch: prepare_dataset(processor, batch), load_from_cache_file=False)
    test_dataset = test_dataset.map(lambda batch: prepare_dataset(processor, batch), remove_columns=test_dataset.column_names, load_from_cache_file=False)

    return train_dataset, test_dataset

# def custom_load_prepared_test_dataset(data_dir, processor: Wav2Vec2Processor, seed=42):
#     data = load_dataset("csv",data_files=data_dir)

#     test_data = data["train"]
    
#     test_data = test_data.map(lambda batch: prepare_dataset(processor, batch))
#     return test_data