import os
from os import path
root_dir = path.dirname(path.dirname(__file__))
import sys
if root_dir not in sys.path:
    sys.path.append(root_dir)
from tqdm import tqdm
from ArgsManager import ArgsManager
import numpy as np
import json

class TextAnalyzer:

    def __init__(self):
        self.EMBEDDING_DIM = -1
        self.VOCAB_SIZE = -1
        self.words_dict ={}
        self.embeddings = []
        self.file_pattern = path.join('glove.6B.{0}d.txt')

    def load_glove_embeddings(self, basedir:str, dim = 100, vocab_size = 100000):
        self.EMBEDDING_DIM = dim
        self.VOCAB_SIZE = vocab_size
        glove_file = path.join(basedir, str.format(self.file_pattern,str(dim)))
        index = 1
        self.words_dict ={}
        self.embeddings = []
        self.words_dict['<PAD>'] = 0
        self.words_dict['<UNK>'] = 1
        self.embeddings.append(np.zeros((self.EMBEDDING_DIM,), dtype='float32'))
        self.embeddings.append(np.zeros((self.EMBEDDING_DIM,), dtype='float32'))
        index = len(self.words_dict)
        with open(glove_file, 'r',encoding="utf8") as f:
            for line in tqdm(f,unit='vector'):
                values = line.split()
                word = values[0]
                self.words_dict[word] = index
                self.embeddings.append( np.asarray(values[1:], dtype='float32'))
                index +=1
                if index > self.VOCAB_SIZE:
                    break
        assert(len(self.words_dict) == len(self.embeddings))
        print('processed %s word embeddings in Glove file.' % len(self.embeddings))

    def save_as(self, save_dir:str, name:str):
        if len(self.words_dict)==0:
            raise Exception('Nothing to save')
        save_dir = path.join(save_dir, name)
        if not path.exists(save_dir):
            os.makedirs(save_dir)
        embeddings = np.array( self.embeddings)
        print('saving data')
        with open(path.join(save_dir, 'word-index.json'), 'w') as f:
            f.write(json.dumps(self.words_dict, indent=2))
        np.save(path.join(save_dir, 'emb.npy'), embeddings)
        print('Done!')
        return save_dir

    def load_master_embeddings(self, base_dir:str):
        with open(path.join( base_dir,'word-index.json'),'r') as f:
            self.words_dict = json.load(f)
        self.embeddings = np.load(path.join(base_dir, 'emb.npy'),allow_pickle=True)
        self.embeddings = list(np.squeeze(self.embeddings))

if __name__ == "__main__":
    
    args = ArgsManager(None,use_app_data= True)
    ta = TextAnalyzer()
    ta.load_glove_embeddings( args['embedding-dir'],vocab_size= args['vocab_size'])
    print(len(ta.words_dict), len(ta.embeddings), np.array(ta.embeddings).shape)
    save_dir = ta.save_as(save_dir = args['embedding-lookup-dir'],name = '100d_100k')
    args.set('active-embedding-lookup-dir', save_dir)
    args.save()
    ta = TextAnalyzer()
    ta.load_master_embeddings(base_dir = args['active-embedding-lookup-dir'])
    print(len(ta.words_dict), len(ta.embeddings), np.array(ta.embeddings).shape)