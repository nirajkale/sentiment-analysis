# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from Tokenizer import BlanketTokenizer
from ArgsManager import ArgsManager

args= ArgsManager(use_app_data= True)
# load model
model = load_model('model.h5')
# summarize model.
model.summary()
# load dataset
data = ["clashes with police."]
pipeline = BlanketTokenizer(args)
data = pipeline.parse_texts(data)
print(data)

sentiment = model.predict(data,verbose = 0)[0]
print(sentiment[0])

if sentiment[0] > 0.5:
    print('Positive')
else:
    print('Negative')
