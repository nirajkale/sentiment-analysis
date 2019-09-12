import os
from os import path
root_dir = path.dirname(path.dirname(__file__))
import sys
if root_dir not in sys.path:
    sys.path.append(root_dir)
from ArgsManager import ArgsManager
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from TextProcessing import SimpleProcessing
from tqdm import tqdm

class BlanketTokenizer:

    def __init__(self, args:ArgsManager, emb_lookup_dir:str=None, use_active_emb_lookup:bool=True):
        self.args = args
        if use_active_emb_lookup or emb_lookup_dir is None:
            emb_lookup_dir = args['active-embedding-lookup-dir']
        with open(path.join( emb_lookup_dir,'word-index.json'),'r') as f:
            self.word_index = json.load(f)
        self.simple_processing = SimpleProcessing(self.args)
        self.index_unk = self.word_index['<UNK>']
        self.index_pad = self.word_index['<PAD>']
        self.sequence_size = args['sequence_size']
        self.padding_type = args['padding_type']

    def parse_texts(self, texts:[]):
        texts = self.simple_processing.parse_texts(texts)
        sequences = []
        for text in tqdm(texts):
            sequence = text_to_word_sequence(text)
            sequences.append([self.word_index[w] if w in self.word_index else self.index_unk for w in sequence])
        sequences = pad_sequences(sequences, maxlen= self.sequence_size, padding= self.padding_type)
        return sequences

    def parse_text(self, text):
        return self.parse_texts([text])

    def recover_texts(self, sequences):
        index_word = {v: k for k, v in self.word_index.items()}
        for sequence in sequences:
            print(sequence)
            print(' '.join([index_word[i] for i in sequence]))

if __name__ == '__main__':

    sentences = ['I need to change my pincode as soon as possible',
                'I need info on how to update my postal-code',
                'My name is Ravi and i would like to change my address',
                'i dont wanna cancel this name updation',
                'never say never',
                'Need some info on how to do this update for servicing and need to do it quickly']

    args = ArgsManager()

    pipeline = BlanketTokenizer(args, use_active_emb_lookup= True)
    x = pipeline.parse_texts(sentences)
    print(x.shape, x.__class__)
    pipeline.recover_texts(x)