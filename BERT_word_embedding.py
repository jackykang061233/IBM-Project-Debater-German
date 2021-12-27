import torch
# from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")


def concatenate(files):
    all_lines = []
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
        all_lines += lines
    bert_sentences = ["[CLS] " + line + " [SEP]" for line in all_lines]
    return bert_sentences


files = ['/Users/kangchieh/Downloads/wiki_00', '/Users/kangchieh/Downloads/wiki_01']
sentences = concatenate(files)
tokenized_sentences = []
for sentence in sentences:
    tokenized_sentences.append(tokenizer.tokenize(sentence))
print(tokenized_sentences)