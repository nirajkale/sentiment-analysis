from keras.preprocessing.text import Tokenizer
from ArgsManager import ArgsManager
import json
from os import path
import numpy as np
from TextProcessing import SimpleProcessing
from DataChinker import DataChinker
from common import shuffle_arrays_in_unison, save_tokenizer, load_tokenizer, ensure_dir
from keras import layers, models, optimizers ,callbacks, regularizers



if __name__ == "__main__":
    
    args= ArgsManager(use_app_data= True)
    text_pipe = SimpleProcessing( args)
    app_data_dir = app_data_dir = path.join( path.dirname(__file__), 'app_data')
    tokenizer = load_tokenizer(args['processed-data-dir'])
    model = models.load_model(r'app_data\checkpoints\bow_v1\lr_0.0001_e25\01-0.06.hdf5')

    texts = [
        'Pakistan halts trade with India',
        'It\'s hard to not be anxious when nowhere feels safe anymore.'
    ]

    texts = text_pipe.parse_texts(texts)
    data = tokenizer.texts_to_matrix(texts)
    inf = model.predict( data, verbose = True)
    print(inf)
    print('done')



