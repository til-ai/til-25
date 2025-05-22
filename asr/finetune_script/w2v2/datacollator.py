import torch
from transformers import Wav2Vec2Processor

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPaddingAndNoise:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    noise_level: float = 0.005
    
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Adds noise to the waveform."""
        noise = torch.randn_like(waveform) * self.noise_level
        return waveform + noise

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:        
        if "audio" in feature:
            for feature in features:
                input_features = []
                waveform = torch.tensor(feature["audio"]["array"])  # Access raw waveform
                sampling_rate = feature["audio"]["sampling_rate"]  # Extract sampling rate

                # Add noise to the waveform
                noisy_waveform = self.add_noise(waveform)

                inputs = self.processor(noisy_waveform.numpy(), sampling_rate=sampling_rate, return_tensors="pt")
                input_features.append({"input_values": inputs.input_values[0]})
        else:
            input_features = [{"input_values": f["input_values"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
        
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return {
            "input_values": batch["input_values"],
            "labels": batch["labels"],
        }
        
@dataclass
class SimpleCTCCollator:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        return batch
