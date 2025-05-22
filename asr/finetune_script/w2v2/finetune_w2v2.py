from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer

import evaluate
import torch
import random
import pandas as pd
import numpy as np
import re

from dataset_loader import custom_load_prepared_dataset
from datacollator import DataCollatorCTCWithPaddingAndNoise
from generate_vocab_dict import generate_vocab_dict
from processor import create_processor


data_dir = "/home/jupyter/til-25-hihi/asr/finetune_script/w2v2/asr_edited.jsonl"

output_dir = "/home/jupyter/til-25-hihi/asr/finetune_script/w2v2t/wav2vec2_trained_model1"
logging_dir = "/home/jupyter/til-25-hihi/asr/finetune_script/w2v2/w2v2_logs1"

vocab_dict = generate_vocab_dict(data_dir)
print('VOCAB DICT: ', vocab_dict)

processor = create_processor()

train_dataset, test_dataset = custom_load_prepared_dataset(data_dir, processor)

data_collator = DataCollatorCTCWithPaddingAndNoise(
    processor=processor,
    padding=True
)

# wer_metric = evaluate.load("wer")
# def compute_metrics(pred):
#     pred_logits = pred.predictions
#     pred_ids = np.argmax(pred_logits, axis=-1)

#     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

#     pred_str = processor.batch_decode(pred_ids)
#     # we do not want to group tokens when computing the metrics
#     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

#     wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)

#     return {"wer": wer}

# model = Wav2Vec2ForCTC.from_pretrained(
#     "facebook/wav2vec2-large-960h",
#     ctc_loss_reduction="mean", 
#     pad_token_id=processor.tokenizer.pad_token_id,
#     ctc_zero_infinity=True                
# )

# training_args = TrainingArguments(
#     output_dir=output_dir,  # change to a repo name of your choice
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
#     group_by_length=True,
#     logging_dir=logging_dir,
#     learning_rate=1e-5,
#     # warmup_steps=1000,
#     warmup_ratio=0.1,
#     num_train_epochs=3,
#     # max_steps=200,
#     gradient_checkpointing=True,
#     fp16=True,
#     per_device_eval_batch_size=2,
#     save_strategy="steps",
#     save_steps=100,
#     eval_strategy="steps",
#     eval_steps=100,
#     logging_steps=10,
#     save_total_limit=2, 
#     push_to_hub=False,
#     remove_unused_columns=False,
#     weight_decay=0.005,
# )

# trainer = Trainer(
#     args=training_args,
#     model=model,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     processing_class=processor.feature_extractor,
# )

# trainer.train()
# trainer.save_model()

# predictions_aft_finetuning = trainer.predict(test_dataset)
# wer_aft_finetuning = compute_metrics(predictions_aft_finetuning)

# print(f"WER aft fine-tuning: {wer_aft_finetuning['wer']}")
