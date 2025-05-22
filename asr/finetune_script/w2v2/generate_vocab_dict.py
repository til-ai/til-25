import json, os
from dataset_loader import custom_load_dataset

vocab_file_path = 'vocab.json'

### Extract all chars
def extract_all_chars(batch):
    all_text = " ".join(batch["transcript"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def generate_vocab_dict(data_dir):
    # Check if the vocab.json file exists and is not empty
    if  os.path.exists(vocab_file_path) and os.stat(vocab_file_path).st_size > 0: 
        print('loading files...')
        with open(vocab_file_path, 'r') as vocab_file:
            vocab_dict = json.load(vocab_file) 
        return vocab_dict

    else:
        print('generating vocab dict...')
        train_data,test_data = custom_load_dataset(data_dir)
        vocab_train = train_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_data.column_names)
        vocab_test = test_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test_data.column_names)

        ### Combine train test vocab
        vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
        to_remove = {'[',']'}  # Use a set for faster lookup

        filtered_list = [x for x in vocab_list if x not in to_remove]

        vocab_dict = {v: k for k, v in enumerate(sorted(filtered_list))}
        
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        with open(vocab_file_path, 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)
        return vocab_dict
