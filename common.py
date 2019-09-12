import os
from os import path
import numpy as np
from keras import backend as K
import pickle

def ensure_dir(*args, ignore_filename = True):
    dirpath = path.join(*args)
    basedir = dirpath
    if ignore_filename:
        basedir = path.dirname( dirpath)
    if not path.exists(basedir):
        os.makedirs(basedir)
    return dirpath

def shuffle_arrays_in_unison(arrays):
    fixed_length = arrays[0].shape[0]
    for arr in arrays[1:]:
        if arr.shape[0] != fixed_length:
            raise Exception('All the arrays need to have same length')
    shuffled_indices = np.random.permutation(fixed_length)
    for i in range(len(arrays)):
        arrays[i] = arrays[i][shuffled_indices]
    return arrays

def precision(y_true, y_pred): 
	'''
	preceision = TP / (TP + FP)
	'''
	y_pred = K.round(K.clip(y_pred, 0, 1))
	true_positives = K.sum(y_true * y_pred)    # TP
	predicted_positives = K.sum(y_pred)       #TP + FP
	precision = true_positives / (predicted_positives + K.epsilon())  
	return precision

def save_tokenizer(tokenizer, dirpath):
    filepath = ensure_dir(dirpath, 'tokenizer.pickle', ignore_filename= True)
    with open(filepath, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(dirpath):
    filepath = path.join(dirpath, 'tokenizer.pickle')
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)


