import pandas as pd
import re
import json
import string

path_to_text = f'./asr.jsonl'

output_txt = f'./asr_edited.jsonl'


chars_to_remove_regex = r'[\`\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«\_\’\\\—\–]'

### Remove special char
def remove_special_characters(text):
    # remove special characters
    text = re.sub(chars_to_remove_regex, '', text).lower()
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def replace_unk(text):
    text = text.replace("<UNK>","[UNK]").replace("(#)", "[UNK]").replace("*", "[UNK]").replace("<unk>","[UNK]")
    return text

def replace_num(text):
    number_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
        
    return ''.join(number_map.get(char, char) for char in text)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'-', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def correct_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if not line.strip():
                continue  # skip empty lines
            data = json.loads(line)
            if 'transcript' in data:
                data['transcript'] = preprocess_text(data['transcript'])
            outfile.write(json.dumps(data) + '\n')

correct_file(path_to_text, output_txt)
